

import copy
import warnings
from bos_metal import ttnn, op

from tt.modules.ops import FFN
from tt.modules.blocks.attn import MultiheadAttention, CustomMSDeformableAttention

from bevformer.utils import assert_many

__all__ = ['BaseTransformerLayer']

class BaseTransformerLayer(op.BaseModule):
    count = 0
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_func='ttnn.relu'
                 ),
                 operation_order=None,
                 norm_layer=op.LayerNorm,
                 batch_first=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs'
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer, self).__init__()
        self.batch_first = batch_first

        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_layer = norm_layer
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = op.ModuleList()
        
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn'] : # ['self_attn', 'cross_attn']
                attn_cfgs[index].pop('type', None)
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                if operation_name == 'self_attn':
                    attention = MultiheadAttention(**attn_cfgs[index])
                elif operation_name == 'cross_attn':
                    attention = CustomMSDeformableAttention(**attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = op.ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = dict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(FFN(**ffn_cfgs[ffn_index]))

        self.norms = op.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(norm_layer(self.embed_dims, memory_config=ttnn.DRAM_MEMORY_CONFIG))

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        def print_ref_shape():
            ref_pt = kwargs.get('reference_points', None)
            if ref_pt is not None:
                self.cout(f"Reference points: {ref_pt.shape}") 
        print_ref_shape()
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        
        assert_many(query, f'BaseTransformerLayer.input_query.{BaseTransformerLayer.count}')
        identity = query
        query_rank = query.dim()
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, ttnn.Tensor):
            attn_masks = [
                # copy.deepcopy(attn_masks) for _ in range(self.num_attn)
                attn_masks for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
                        
                        
        order = 0
        for layer in self.operation_order:
            if layer == 'self_attn':
                self.cout("Start self_attn")
                print_ref_shape()
                temp_key = temp_value = query
                query = self.attentions[attn_index](     #WARNING: need to pass kwargs
                    query=query,
                    key=temp_key,
                    value=temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query
                self.cout(f"Output of self_attn: {query.shape} {query.dtype}")
                print_ref_shape()

            elif layer == 'norm':
                self.cout("Start norm")
                print_ref_shape()
                query = self.norms[norm_index](query)
                norm_index += 1
                self.cout(f"Output of norm: {query.shape} {query.dtype}")
                print_ref_shape()

            elif layer == 'cross_attn':
                self.cout("Start cross_attn")
                print_ref_shape()
                # print(f"Input query: {query.shape} {query.dtype}")
                query = self.attentions[attn_index](    #WARNING: need to pass kwargs
                    query=query,
                    key=key,
                    value=value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query
                self.cout(f"Output of cross_attn: {query.shape}")
                print_ref_shape()

            elif layer == 'ffn':
                self.cout("Start ffn")
                print_ref_shape()
                query = self.ffns[ffn_index](
                    query, 
                    identity if self.pre_norm else None
                )
                ffn_index += 1
                if query_rank == 3:
                    query = query.squeeze(0)
                self.cout(f"Output of FFN: {query.shape}")
                print_ref_shape()
            
            assert_many(query, f'BaseTransformerLayer.{layer}.{order}.{BaseTransformerLayer.count}')
            order += 1
            BaseTransformerLayer.count += 1
        return query # ttnn.reallocate(query, device=self.device)