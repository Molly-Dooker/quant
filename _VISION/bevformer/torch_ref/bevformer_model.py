from .bevformer_tiny_cfg import bevformer_tiny_cfg, test_bev_tiny_cfg
from .modules.utils import build_detector


BEVFormer = build_detector(bevformer_tiny_cfg, test_cfg=test_bev_tiny_cfg)