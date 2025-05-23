from .builder import Registry, build_model_from_cfg

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
PLUGIN_LAYERS = Registry('plugin layer')

DROPOUT_LAYERS = Registry('drop out layers')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')

TRANSFORMER = Registry('Transformer')
ACTIVATION_LAYERS = Registry('activation layer')

LOSSES = Registry('loss')

NECKS = Registry('necks')
BACKBONES = Registry('necks')
HEADS = Registry('heads')
DETECTORS = Registry('detectors')
MODELS = Registry('model', build_func=build_model_from_cfg)
LINEAR_LAYERS = Registry('linear layers')
DROPOUT_LAYERS = Registry('drop out layers')
#
BBOX_CODERS = Registry('bbox coder')
BBOX_ASSIGNERS = Registry('bbox assigner')
BBOX_SAMPLERS = Registry('bbox sampler')


