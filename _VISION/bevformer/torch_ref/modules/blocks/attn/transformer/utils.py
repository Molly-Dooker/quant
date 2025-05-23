from bevformer.torch_ref.modules.utils import build_from_cfg
from bevformer.torch_ref.modules.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)