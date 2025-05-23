import torch
import ttnn
from bos_metal import op
from torchvision.transforms.functional import rotate
from bevformer.utils import assert_many, load_many

from tt.modules.bev import BEVFormerEncoder, DetectionTransformerDecoder

__all__ = ['PerceptionTransformer']

class PerceptionTransformer(op.BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 encoder, 
                 decoder, #NOTE: not in used for inference yet
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        assert encoder is not None and decoder is not None, "encoder and decoder must be given"
        self.encoder = BEVFormerEncoder(**encoder)
        self.decoder = DetectionTransformerDecoder(**decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = op.BaseParameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims),
            map_func=lambda x: ttnn.from_torch(x, device=self.device)                                     
        )
        self.reference_points = op.Linear(self.embed_dims, 3, dtype=ttnn.bfloat16)
        self.cams_embeds = op.BaseParameter(
            torch.Tensor(self.num_cams, self.embed_dims),
            map_func=lambda x: ttnn.from_torch(x, device=self.device)            
        )
        self.can_bus_mlp = op.Sequential(
            op.Linear(18, self.embed_dims // 2),
            op.Functional(ttnn.relu),
            op.Linear(self.embed_dims // 2, self.embed_dims),
            op.Functional(ttnn.relu),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', op.LayerNorm(self.embed_dims))

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            img_metas=None):
        """
        obtain bev features.
        """
        #NOTE: hide for fast debug
        bs = mlvl_feats[0].size(0)
        bev_queries = ttnn.repeat_interleave(
            bev_queries.to_layout(ttnn.ROW_MAJOR_LAYOUT).unsqueeze(1), bs, 1
        )
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        # assert_many(ttnn.to_torch_all(bev_pos), "perception_transformer.encoder.bev_pos")

        # obtain rotation angle and shift with ego motion
        print("Extract rotation angle and shift")
        delta_x = ttnn.concat([each['can_bus'][0] for each in img_metas], 0).tile()
        delta_y = ttnn.concat([each['can_bus'][1] for each in img_metas], 0).tile()
        can_bus = img_metas[0]['can_bus'][-2]
        ego_angle = ttnn.concat([
            (each['can_bus'][-2].tile() / torch.pi) * 180 for each in img_metas
        ], 0)
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = ttnn.sqrt(
            ttnn.square(delta_x.tile()) + 
            ttnn.square(delta_y.tile())
        )
        delta = ttnn.atan2(delta_y, delta_x) 
        delta_x.deallocate()
        delta_y.deallocate()
        translation_angle = (delta / torch.pi) * 180
        bev_angle = ego_angle - translation_angle
        ego_angle.deallocate()
        translation_angle.deallocate()
        shift_y = translation_length * \
            (ttnn.cos(bev_angle / 180 * torch.pi) / grid_length_y) / bev_h
        shift_x = translation_length * \
            (ttnn.sin(bev_angle / 180 * torch.pi) / grid_length_x) / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = ttnn.concat([shift_x, shift_y], 0).permute(1, 0)  # xy, bs -> bs, xy
        shift_x.deallocate()
        shift_y.deallocate()
        # assert_many(ttnn.to_torch_all(shift), "perception_transformer.encoder.shift")

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].\
                        reshape(bev_h, bev_w, -1).\
                        permute(2, 0, 1)
                    #TODO: FALLBACK
                    tmp_prev_bev = rotate(tmp_prev_bev.torch(), 
                                          rotation_angle.torch(),
                                          center=self.rotate_center.torch())
                    tmp_prev_bev = ttnn.from_torch(tmp_prev_bev)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
                    tmp_prev_bev.deallocate()
            # assert_many(ttnn.to_torch_all(prev_bev), "perception_transformer.encoder.prev_bev")

        # add can bus signals
        can_bus = ttnn.concat([each['can_bus'] for each in img_metas])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus).squeeze(0) # 4D outputs
        can_bus = ttnn.to_memory_config(can_bus, ttnn.DRAM_MEMORY_CONFIG)
        # assert_many(ttnn.to_torch_all(bev_queries), "perception_transformer.encoder.bev_queries_before")
        # assert_many(ttnn.to_torch_all(can_bus), "perception_transformer.encoder.can_bus")
        
        # can_bus = ttnn.to_layout(can_bus, ttnn.TILE_LAYOUT)
        bev_queries = ttnn.to_layout(bev_queries, ttnn.TILE_LAYOUT)
        bev_queries = bev_queries + can_bus if self.use_can_bus else 0
        can_bus.deallocate()
        # TODO: test failed
        # assert_many(ttnn.to_torch_all(bev_queries), "perception_transformer.encoder.bev_queries")

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # assert_many(ttnn.to_torch_all(feat), "perception_transformer.encoder.feat")
            if self.use_cams_embeds:
                # TODO: check output shape of this is matched
                cams_embeds = self.cams_embeds.ttnn_data.unsqueeze(1, 2)
                # assert_many(ttnn.to_torch_all(cams_embeds), "perception_transformer.encoder.cams_embeds")
                ##
                cams_embeds = ttnn.to_layout(cams_embeds, ttnn.TILE_LAYOUT)
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                feat = feat + cams_embeds
                ttnn.deallocate(cams_embeds)
                
                # feat = op.Add(deallocate_input=True)(feat, cams_embeds)
                # assert_many(ttnn.to_torch_all(feat), "perception_transformer.encoder.feat_with_cams_embeds")
            feat = feat + self.level_embeds.ttnn_data.unsqueeze(0, 0)[:, :, lvl:lvl + 1, :].tile()
            # assert_many(ttnn.to_torch_all(feat), "perception_transformer.encoder.feat_with_level_embeds")
            spatial_shapes.append(torch.Tensor([spatial_shape]))
            feat_flatten.append(feat.to_layout(ttnn.ROW_MAJOR_LAYOUT))
            # assert_many(ttnn.to_torch_all(feat_flatten), "perception_transformer.encoder.feat_flatten")
            #NOTE: to ROLW_MAJOR_LAYOUT to avoid concat global CB alloc error
        feat_flatten = ttnn.concat(feat_flatten, 2)
        # assert_many(ttnn.to_torch_all(feat_flatten), "perception_transformer.encoder.feat_flatten_concat")
        spatial_shapes = torch.Tensor([spatial_shape]) # ttnn.concat(spatial_shapes, 0)
        # TODO: FALLBACK
        
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # assert_many(ttnn.to_torch_all(spatial_shapes), "perception_transformer.encoder.spatial_shapes")
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        # assert_many(ttnn.to_torch_all(feat_flatten), "perception_transformer.encoder.feat_flatten")

        # bev_queries = load_many("perception_transformer.encoder.bev_queries")
        # bev_pos = load_many("perception_transformer.encoder.bev_pos")
        # feat_flatten = load_many("perception_transformer.encoder.feat_flatten")
        # spatial_shapes = load_many("perception_transformer.encoder.spatial_shapes")
        # level_start_index = load_many("perception_transformer.encoder.level_start_index")
        
        # shift = load_many("perception_transformer.encoder.shift")
        # prev_bev = None
        print("Running PerceptionTransformer encoder")
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=ttnn.from_torch_all(bev_pos),
            spatial_shapes=ttnn.from_torch(spatial_shapes, 
                                           memory_config=ttnn.L1_MEMORY_CONFIG,
                                           dtype=ttnn.bfloat16,
                                           device=self.device),
            level_start_index=ttnn.from_torch(level_start_index,
                                              memory_config=ttnn.L1_MEMORY_CONFIG,
                                              dtype=ttnn.uint8,
                                              device=self.device),
            prev_bev=prev_bev,
            shift=ttnn.from_torch_all(shift),
            img_metas=img_metas,
        )

        return bev_embed
    
    def get_states_and_refs(
        self,
        bev_embed,
        object_query_embed,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None
    ):
        bs = bev_embed.shape[1]
        num_splits = object_query_embed.size(1) // self.embed_dims  # 2
        query_chunks = ttnn.split(object_query_embed, num_splits, dim=1)
        query_pos, query = query_chunks[0], query_chunks[1]
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        assert_many(reference_points, "perception_transformer.reference_points")
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.tile().sigmoid()

        init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        print("Running PerceptionTransformer decoder")
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=ttnn.as_tensor([[bev_h, bev_w]], dtype=ttnn.bfloat16, device=query.device(), memory_config=ttnn.L1_MEMORY_CONFIG),
            level_start_index=ttnn.as_tensor([0], dtype=ttnn.uint8, device=query.device(), memory_config=ttnn.L1_MEMORY_CONFIG),
            img_metas=img_metas
        )
        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out
    
    
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  
        assert_many(bev_embed, "perception_transformer.fw.bev_embed")

        # bev_embed = load_many("perception_transformer.fw.bev_embed")
        # bev_embed = ttnn.from_torch(bev_embed, device=self.device, dtype=ttnn.bfloat16)
        
        bs = mlvl_feats[0].shape[0]
        # Deallocate for free L1 space
        # for lvl in range(len(mlvl_feats)):
            # ttnn.deallocate(mlvl_feats[lvl])
        query_pos = object_query_embed.ttnn_data[:, 0:self.embed_dims]
        assert_many(query_pos, "perception_transformer.fw.query_pos")
        query = object_query_embed.ttnn_data[:, self.embed_dims:]
        assert_many(query, "perception_transformer.fw.query")
        # 
        query_pos = ttnn.expand(ttnn.unsqueeze(query_pos, 0), bs, -1, -1) # .unsqueeze(0).expand(bs, -1, -1)
        # 
        query = ttnn.expand(ttnn.unsqueeze(query, 0), bs, -1, -1)
        #
        reference_points = self.reference_points(query_pos)
        # assert_many(reference_points.squeeze(0), "perception_transformer.fw.reference_points")
        #
        reference_points = ttnn.sigmoid_accurate(reference_points)
        # assert_many(reference_points.squeeze(0), "perception_transformer.fw.reference_points_sigmoid")
        reference_points = ttnn.squeeze(reference_points, 0)
        #
        init_reference_out = ttnn.clone(reference_points, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = ttnn.to_memory_config(bev_embed, ttnn.DRAM_MEMORY_CONFIG)
        bev_embed = bev_embed.permute(1, 0, 2)
        # bev_embed = ttnn.reallocate(bev_embed, memory_config=ttnn.L1_MEMORY_CONFIG)
        # bev_embed = ttnn.to_memory_config(bev_embed, ttnn.L1_MEMORY_CONFIG)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=ttnn.as_tensor([[bev_h, bev_w]], device=query.device(), dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
            level_start_index=ttnn.as_tensor([0], dtype=ttnn.uint8, device=query.device(), memory_config=ttnn.L1_MEMORY_CONFIG),
            **kwargs)
        
        # inter_states = ttnn.from_torch_all(load_many("perception_transformer.fw.inter_states"), layout= ttnn.TILE_LAYOUT, device=self.device)
        # inter_references = ttnn.from_torch_all(load_many("perception_transformer.fw.inter_references"), layout= ttnn.TILE_LAYOUT, device=self.device)
        assert_many(bev_embed, "perception_transformer.fw.bev_embed")                            
        assert_many(inter_states, "perception_transformer.fw.inter_states")
        assert_many(init_reference_out, "perception_transformer.fw.init_reference_out")
        assert_many(inter_references, "perception_transformer.fw.inter_references")
        
        inter_references_out = inter_references
        init_reference_out = ttnn.to_memory_config(init_reference_out, ttnn.L1_MEMORY_CONFIG)
        return bev_embed, inter_states, init_reference_out, inter_references_out
