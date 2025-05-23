import torch.nn as nn
from torch import Tensor as Tensor
from modules.backbone import ResNet50
from modules.neck import FPN
from modules.registry import build_from_cfg
from modules.registry import NECKS
from modules.bev import BEVFormer

class BEVFormerTorch(BEVFormer):
    def __init__(self, 
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False):
        super(BEVFormerTorch, self).__init__(use_grid_mask=use_grid_mask,
                                                pts_voxel_layer=pts_voxel_layer,
                                                pts_voxel_encoder=pts_voxel_encoder,
                                                pts_middle_encoder=pts_middle_encoder,
                                                pts_fusion_layer=pts_fusion_layer,
                                                img_backbone=img_backbone,
                                                pts_backbone=pts_backbone,
                                                img_neck=img_neck,
                                                pts_neck=pts_neck,
                                                pts_bbox_head=pts_bbox_head,
                                                img_roi_head=img_roi_head,
                                                img_rpn_head=img_rpn_head,
                                                train_cfg=train_cfg,
                                                test_cfg=test_cfg,
                                                pretrained=pretrained,
                                                video_test_mode=video_test_mode)
        
        # Backbone (ResNet-50)
        # self.img_backbone = ResNet50()
        # # Neck (FPN)
        # img_neck_cfg = dict(type='FPN',
        #               in_channels=[2048],
        #               out_channels=256,
        #               start_level=0,
        #               add_extra_convs='on_output',
        #               num_outs=1,
        #               relu_before_extra_convs=True)
        
        # self.img_neck = build_from_cfg(img_neck_cfg, NECKS)


    # def forward(self, images):
    #     """
    #     Args:
    #         images: Tensor of shape [B, 3, H, W], batch of input images.
    #     Returns:
    #         cls_preds: Classification predictions [B, num_queries, num_classes].
    #         bbox_preds: Bounding box predictions [B, num_queries, 4].
    #     """
    #     # Backbone forward
    #     features = self.img_backbone(images)
    #     # # FPN
    #     features = self.img_neck(features)
        
        
    #     # # Flatten spatial dimensions for transformer
    #     # B, C, H, W = features.size()
    #     # bev_input = features.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        
    #     # # Add BEV spatial embedding
    #     # bev_input += self.bev_embedding.unsqueeze(1)
        
    #     # # Transformer encoding
    #     # bev_output = self.bev_encoder(bev_input)  # [H*W, B, C]
        
    #     # # Query embeddings interaction
    #     # queries = self.query_embeddings.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
    #     # query_output = self.bev_encoder(queries, src_key_padding_mask=None)  # [num_queries, B, C]
        
    #     # # Detection head
    #     # cls_preds = self.cls_head(query_output)  # [num_queries, B, num_classes]
    #     # bbox_preds = self.bbox_head(query_output)  # [num_queries, B, 4]
        
    #     # return cls_preds.permute(1, 0, 2), bbox_preds.permute(1, 0, 2)  # [B, num_queries, num_classes], [B, num_queries, 4]
    #     return features
    
    