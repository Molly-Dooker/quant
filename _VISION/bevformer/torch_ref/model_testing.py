import random
import torch
import numpy as np

from modules.utils import build_detector





point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3
# Define constants for NuScenes-like data
BATCH_SIZE = 2
SEQUENCE_LENGTH = 3  # Number of frames in a queue
NUM_CAMERAS = 6      # Number of cameras
CHANNELS = 3         # Image channels (RGB)
HEIGHT = 900         # Image height
WIDTH = 1600          # Image width
CAN_BUS_SIZE = 18    # Size of the can_bus signal


def set_random_seed(seed=42):
    """Ensures reproducibility across different machines."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
# Mock Box3D class (replace with actual implementation)
class Box3D:
    def __init__(self, bboxes, code_size):
        self.bboxes = torch.tensor(bboxes, dtype=torch.float32)
        self.code_size = code_size

    def to(self, device):
        self.bboxes = self.bboxes.to(device)
        return self

    def __call__(self, bboxes, code_size):
        return Box3D(bboxes, code_size)
    
    
# Example input preparation for forward_test
def create_forward_test_inputs():
    set_random_seed(42)  # Ensure deterministic results

    # Generate example image data (CPU-based, converted to correct device later)
    img = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, NUM_CAMERAS, CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)

    # Generate example lidar2img matrices (NumPy -> Torch tensor)
    lidar2img = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH, NUM_CAMERAS, 4, 4).astype(np.float32)
    
    # Convert to torch tensor to avoid precision differences
    lidar2img_torch = torch.from_numpy(lidar2img)

    # Generate example CAN bus data (NumPy array)
    can_bus = np.arange(CAN_BUS_SIZE, dtype=np.float32)

    # Generate metadata with consistent data types
    img_metas = [
        [
            {
                'scene_token': f'scene_{batch_idx + 1}',
                'prev_bev_exists': True,
                'can_bus': can_bus,  # NumPy ndarray
                'img_shape': [[HEIGHT, WIDTH]],  # Keep nested list format
                'scale_factor': 1.0,
                'flip': False,
                'lidar2img': lidar2img_torch[batch_idx, frame_idx].tolist(),  # Maintain float32 precision
                'box_type_3d': Box3D,  # Placeholder for actual 3D box type
            }
            for frame_idx in range(SEQUENCE_LENGTH)
        ]
        for batch_idx in range(BATCH_SIZE)
    ]

    return img, img_metas


if __name__ == '__main__':
    cfg = dict(
        type='BEVFormer',
        use_grid_mask=True,
        video_test_mode=True,
        pretrained=dict(img='torchvision://resnet50'),
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'),
        img_neck=dict(
            type='FPN',
            in_channels=[2048],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=_num_levels_,
            relu_before_extra_convs=True),
        pts_bbox_head=dict(
            type='BEVFormerHead',
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=9,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type='PerceptionTransformer',
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                encoder=dict(
                    type='BEVFormerEncoder',
                    num_layers=3,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=_dim_,
                                    num_points=8,
                                    num_levels=_num_levels_),
                                embed_dims=_dim_,
                            )
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm'))),
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                        ],

                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm')))),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
                )))

   
    test_cfg= dict(pts=dict(use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))

    
    model = build_detector(cfg, test_cfg=test_cfg)

    model.eval()
    state_dict = torch.load('./bevformer/torch_ref/ckpts/bevformer_tiny_epoch_24.pth',
                            map_location='cpu')["state_dict"]

    load_result =  model.load_state_dict(state_dict=state_dict, 
                                        strict=False)
    # Inspect missing and unexpected keys
    
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
    # Prepare example inputs
    img, img_metas  = create_forward_test_inputs()
    # print(img)
    # print(img_metas)
    output = model.forward_test(img_metas, img)
    print(output)
