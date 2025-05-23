import copy
import torch
from bos_metal import ttnn, op

from bevformer.utils import multi_apply, inverse_sigmoid
from tt.modules.bbox import NMSFreeCoder
from .detr_head import DETRHead

from bevformer.utils import assert_many

class BEVFormerHead(DETRHead):
    count = 0
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 past_steps=4,
                 fut_steps=4,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.bbox_coder = NMSFreeCoder(**bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_h = self.pc_range[3] - self.pc_range[0]
        self.real_w = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = op.BaseParameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.adder = op.Add()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        def init_cls(self):
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(op.Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(op.LayerNorm(self.embed_dims))
                cls_branch.append(op.Functional(ttnn.relu))
            cls_branch.append(op.Linear(self.embed_dims, self.cls_out_channels))
            fc_cls = op.Sequential(*cls_branch)
            return fc_cls
        
        def init_reg(self):
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(op.Linear(self.embed_dims, 
                                            self.embed_dims))
                reg_branch.append(op.Functional(ttnn.relu))
            reg_branch.append(op.Linear(self.embed_dims, self.code_size))
            reg_branch = op.Sequential(*reg_branch)
            return reg_branch

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = op.ModuleList([init_cls(self) for i in range(num_pred)])  # _get_clones(fc_cls, num_pred)
            self.reg_branches = op.ModuleList([init_reg(self) for i in range(num_pred)]) # _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = op.ModuleList(
                [init_cls(self) for _ in range(num_pred)])
            self.reg_branches = op.ModuleList(
                [init_reg(self) for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = op.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = op.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        object_query_embeds = self.query_embedding.weight
        assert_many(object_query_embeds, 'BEVFormerHead.object_query_embeds')
        bev_queries = self.bev_embedding.weight.ttnn_data
        assert_many(bev_queries, 'BEVFormerHead.bev_queries')

        bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), memory_config=ttnn.L1_MEMORY_CONFIG) 
        bev_pos = self.positional_encoding(bev_mask)
        assert_many(bev_pos, 'BEVFormerHead.bev_pos')

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer.forward(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )
        
        assert_many(outputs, f'BEVFormerHead.outputs.{BEVFormerHead.count}')

        bev_embed, hs, init_reference, inter_references = outputs
        # inter_references = inter_references.row_major()
        # init_reference = init_reference.row_major()
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        order = 0
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            assert_many(reference, f'BEVFormerHead.reference.{BEVFormerHead.count}.{lvl}')
            reference = ttnn.inverse_sigmoid(reference)
            assert_many(reference, f'BEVFormerHead.reference_sigmoid.{BEVFormerHead.count}.{lvl}')
            outputs_class = self.cls_branches[lvl](hs[lvl:lvl+1])
            # if lvl == 3: return hs[lvl:lvl+1]
            tmp = self.reg_branches[lvl](hs[lvl:lvl+1]).squeeze(0)  # shape: ([1, B, num_q, 10])
            assert reference.shape[-1] == 3
            # tmp = ttnn.to_layout(tmp, ttnn.ROW_MAJOR_LAYOUT)
            # reference = ttnn.to_layout(reference, ttnn.ROW_MAJOR_LAYOUT)
            tmp = ttnn.permute(tmp, (2, 0, 1))
            reference = ttnn.permute(reference, (2, 0, 1))
            t1 = self.adder(tmp[0:2], reference[0:2])
            t1 = t1.sigmoid_()
            t2 = self.adder(tmp[4:5], reference[2:3])
            t2 = t2.sigmoid_()
            ttnn.deallocate(reference)
            t3 = (t1[0:1]* (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            t4 = (t1[1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            t5 = tmp[2:4]
            t7 = tmp[5:]
            t6 = (t2[0:1] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            tmp = ttnn.concat([t3, t4, t5, t6, t7], 0)
            tmp = ttnn.reallocate(tmp)
            tmp = ttnn.permute(tmp, (1, 2, 0))
            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            order += 1

        outputs_classes = ttnn.stack(outputs_classes, 0).squeeze(1)
        outputs_coords = ttnn.stack(outputs_coords, 0).squeeze(2)
        
        assert_many(outputs_classes, f'BEVFormerHead.outputs_classes.{BEVFormerHead.count}')
        assert_many(outputs_coords, f'BEVFormerHead.outputs_coords.{BEVFormerHead.count}')
        assert_many(bev_embed, f'BEVFormerHead.bev_embed.{BEVFormerHead.count}')

        ttnn.deallocate(hs)
        ttnn.deallocate(init_reference)
        ttnn.deallocate(inter_references)
        
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        BEVFormerHead.count += 1
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)


    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
