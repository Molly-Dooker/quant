import copy
from bos_metal import op # device_box, profiler, 
from tt.modules.bev.bevformer_layer import BEVFormerLayer
from tt.modules.blocks.transformer.detr_transformer_decoder_layer import DetrTransformerDecoderLayer

from bevformer.utils import assert_many

__all__ = ['TransformerLayerSequence']

class TransformerLayerSequence(op.BaseModule):
    count = 0
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None, **kwargs):
        super(TransformerLayerSequence, self).__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = op.ModuleList()
        # 
        # 
        for i in range(num_layers):
            transformer_cls = transformerlayers[i].pop('type')
            if transformer_cls == 'BEVFormerLayer':
                self.layers.append(BEVFormerLayer(**transformerlayers[i]))
            elif transformer_cls == 'DetrTransformerDecoderLayer':
                self.layers.append(DetrTransformerDecoderLayer(**transformerlayers[i]))
                
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        for layer in self.layers:
            assert_many(query, f'TransformerLayerSequence.{TransformerLayerSequence.count}.input_query')
            assert_many(query, f'TransformerLayerSequence.{TransformerLayerSequence.count}.input_key')
            assert_many(query, f'TransformerLayerSequence.{TransformerLayerSequence.count}.input_value')
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            assert_many(query, f'TransformerLayerSequence.{TransformerLayerSequence.count}.output_query')
        return query
