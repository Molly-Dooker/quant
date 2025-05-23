from bevformer.torch_ref.modules.utils import build_from_cfg
from bevformer.torch_ref.modules.registry import FEEDFORWARD_NETWORK

def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)