import numpy as np
import random
import torch
import ttnn
from bos_metal import device_box, profiler
import bos_metal as bm
from tt.modules.backbone.resnet import ResNet
from tt.modules.neck.fpn import FPN
from tt.modules.head.bevformer_head import BEVFormerHead
from tt.modules.bev import BEVFormer as BEVFormerTTNN
from torch.nn import functional as F
from pathlib import Path

from bevformer.bevformer_config_ttnn import bevformer_config_v55_4x5
from bevformer.bevformer_config_split import bevformer_config_split_v55_4x5


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
BATCH_SIZE = 1       # Number of samples in a batch
SEQUENCE_LENGTH = 1  # Number of frames in a queue
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
                'lidar2img': [img.numpy() for img in lidar2img_torch[batch_idx, frame_idx]],  # Maintain float32 precision
                'box_type_3d': Box3D,  # Placeholder for actual 3D box type
            }
            for frame_idx in range(SEQUENCE_LENGTH)
        ]
        for batch_idx in range(BATCH_SIZE)
    ]

    return img, img_metas



def load_input(path, location='cpu'):
    input_folder = Path("bevformer/inputs/")
    return torch.load(input_folder / path, location)

def load_inputs(location='cpu'):
        """Load inputs for debugging.
        """
        return {
            'img': load_input('img.pth', location),
            'img_metas': load_input('img_metas.pth', location),
        }


# from bos_models.bevformer.ttnn.utils import save_tensor, load_tensor

bm.RuntimeOptions.enable_verbose()
# bm.RuntimeOptions.enable_golden_test()
# bm.RuntimeOptions.enable_log_bhrc()

pts_bbox_head_cfg=dict(
            # type='BEVFormerHead',
                bev_h=bev_h_,
                bev_w=bev_w_,
                num_query=900,
                num_classes=9,
                in_channels=_dim_,
                sync_cls_avg_factor=True,
                with_box_refine=True,
                as_two_stage=False,
                transformer=dict(
                    # type='PerceptionTransformer',
                    rotate_prev_bev=True,
                    use_shift=True,
                    use_can_bus=True,
                    embed_dims=_dim_,
                    encoder=dict(
                        # type='BEVFormerEncoder',
                        num_layers=3,
                        pc_range=point_cloud_range,
                        bev_h=bev_h_,
                        bev_w=bev_w_,
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
                                        # type='MSDeformableAttention3D',
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
                        # type='DetectionTransformerDecoder',
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
                    # type='NMSFreeCoder',
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    pc_range=point_cloud_range,
                    max_num=300,
                    voxel_size=voxel_size,
                    num_classes=10),
                positional_encoding=dict(
                    # type='LearnedPositionalEncoding',
                    num_feats=_pos_dim_,
                    row_num_embed=bev_h_,
                    col_num_embed=bev_w_,
                    )
    )

# from nuscenes.nuscenes import NuScenes
# import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Test BEVFormer
    device_box.device_config.update({"device_id": 0, "l1_small_size": 50*32*32})
    device = device_box.get()
    
    img_backbone = ResNet(depth=50,
                          num_stages=4,
                          out_indices=[3],
                          )
    img_neck = FPN(in_channels=[2048],
                   out_channels=256,
                   start_level=0,
                   add_extra_convs='on_output',
                   num_outs=1,
                   relu_before_extra_convs=True)
    pts_bbox_head_cfg.pop('type', None)
    pts_bbox_head = BEVFormerHead(**pts_bbox_head_cfg)
    use_grid_mask = False
    
    
    ttnn_model = BEVFormerTTNN(img_backbone, img_neck, pts_bbox_head, False, device=device)
    
    # print(ttnn_model)
    

    inputs_dict = load_inputs()
    img, img_metas  = inputs_dict['img'], inputs_dict['img_metas']
    img = F.interpolate(img[0].squeeze(0), size=(HEIGHT, WIDTH), mode='bilinear', align_corners=False)
    img = [img.unsqueeze(0)]
    img_metas = [img_metas]
    img_metas[0][0]['img_shape'] = [(HEIGHT, WIDTH)]
    img_metas[0][0]['box_type_3d'] = Box3D
    state_dict_for_ttnn = torch.load("./state_dict/bevformer_tiny_ttnn.pth")
    load_result = ttnn_model.load_state_dict(state_dict_for_ttnn, strict=False)   
    # print(ttnn_model)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
    ttnn_model.load_config_dict(bevformer_config_v55_4x5)
    # Split config for running sequence images
    # ttnn_model.load_config_dict(bevformer_config_split_v55_4x5)
    ttnn_out = ttnn_model.forward_test(img=img, img_metas=img_metas)
    print("TTNN output:", ttnn_out)
    device_box.close()
