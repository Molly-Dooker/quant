# from bos_metal import  op
import copy
from bos_metal.core import BaseModule
import ttnn
from bevformer.utils import Validator, bbox3d2result, load_many, assert_many


__all__ = ['BEVFormer']


class BEVFormer(BaseModule):
    def __init__(self, img_backbone, img_neck=None, pts_bbox_head = None, use_grid_mask=False, device=None):
        super(BEVFormer, self).__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.pts_bbox_head = pts_bbox_head
        self.use_grid_mask = use_grid_mask
        self.with_img_neck = img_neck is not None
        self.device=device
        
            # temporal
        self.video_test_mode = False # video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
                
                
            img = ttnn.from_torch(img, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT) # , memory_config=ttnn.L1_MEMORY_CONFIG
            img = ttnn.permute(img, (0, 2, 3, 1)) 
            # Running backbone
            print("Running img_backbone")
            img_feats = self.img_backbone(img)
            # assert_many(img_feats.permute(0, 3, 1, 2), "img_backbone") # ttnn.to_torch(img_feats).permute(0, 3, 1, 2)
            # if isinstance(img_feats, dict):
            #     img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            print("Running img_neck")
            img_feats = self.img_neck(img_feats)
            # assert_many(img_feats[0].permute(0, 3, 1, 2), "img_neck")
            # Loading tensor for debugging
            # img_feats = [ttnn.from_torch(load_many("img_neck").permute(0, 2, 3, 1),
            #                         device=self.device,
            #                         dtype=ttnn.bfloat16)]

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats


    def forward(self, return_loss=False, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None
       
        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = [0,0,0]

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        return bbox_results
    

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
            """Test function"""
            # lidar2img = img_metas[0]['lidar2img']
            img_metas = ttnn.from_torch_all(img_metas, device=self.device)
            # img_metas[0]['lidar2img'] = lidar2img
   
            outs = self.pts_bbox_head.forward(x, img_metas, prev_bev=prev_bev)
            all_cls_scores = ttnn.to_torch(outs['all_cls_scores'])
            all_bbox_preds = ttnn.to_torch(outs['all_bbox_preds'])
            ttnn.deallocate(outs['all_cls_scores'])
            ttnn.deallocate(outs['all_bbox_preds'])
            # ttnn.deallocate(outs['bev_embed'])
            outs['all_cls_scores'] = all_cls_scores
            outs['all_bbox_preds'] = all_bbox_preds
                        
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        img_feats[0] = img_feats[0].row_major()

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
