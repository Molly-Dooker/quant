import warnings
import torch.nn as nn
from bos_metal import ttnn, op, device_box
from bevformer.utils import assert_many, load_many

__all__ = ['MultiheadAttention']


class MultiheadAttention(op.BaseModule):
    count = 0
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = not batch_first

        self.attn = op.MultiheadAttention_(embed_dims, num_heads, **kwargs)
        self.add = op.Add(deallocate_input=False, inplace=False)
        self.add_ = op.Add(deallocate_input=False, inplace=True)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = ttnn.clone(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
                    
        assert_many(query, f'MultiheadAttention.{MultiheadAttention.count}.input_query')
        assert_many(key, f'MultiheadAttention.{MultiheadAttention.count}.input_key')
        assert_many(value, f'MultiheadAttention.{MultiheadAttention.count}.input_value')
        assert_many(identity, f'MultiheadAttention.{MultiheadAttention.count}.identity')
        
        # query = load_many(f'MultiheadAttention.{MultiheadAttention.count}.input_query')
        # key = load_many(f'MultiheadAttention.{MultiheadAttention.count}.input_key')
        # value = load_many(f'MultiheadAttention.{MultiheadAttention.count}.input_value')
        # query_pos = load_many(f'MultiheadAttention.{MultiheadAttention.count}.query_pos')
        # key_pos = load_many(f'MultiheadAttention.{MultiheadAttention.count}.key_pos')
        
        if query_pos is not None:
            query = self.add(query, query_pos)
        if key_pos is not None:
            key = self.add_(key, key_pos)   #inplace
            
        assert_many(query, f'MultiheadAttention.{MultiheadAttention.count}.query_with_pos')
        assert_many(key, f'MultiheadAttention.{MultiheadAttention.count}.key_with_pos')

        # Because the dataflow('key', 'query', 'value') of
        # ``op.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        # query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        if self.batch_first:
            #TODO: ttnn transpose op?
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        print("Running MultiheadAttention")
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        # 1, 900, 256 -> 900, 1, 256
        out = out.permute(1, 0, 2)
        # assert_many(out.squeeze(0), f'MultiheadAttention.{MultiheadAttention.count}.out_attn')
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)      
        assert_many(out, f'MultiheadAttention.{MultiheadAttention.count}.out_attn')
        assert_many(identity, f'MultiheadAttention.{MultiheadAttention.count}.identity')
        
        # identity = ttnn.to_layout(identity, ttnn.ROW_MAJOR_LAYOUT)
        # out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        # assert_many(self.add(identity, out), f'BaseTransformerLayer.self_attn.0.6')
            
        MultiheadAttention.count += 1
        # identity = ttnn.to_memory_config(identity, ttnn.L1_MEMORY_CONFIG)
        return self.add(out, identity)
