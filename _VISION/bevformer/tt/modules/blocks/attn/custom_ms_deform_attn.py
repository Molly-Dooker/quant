import torch
import ttnn
import ttnn.torch_tracer
from bos_metal.core import BaseModule
from bos_metal import device_box, op
from tt.modules.blocks.attn.ms_deform_attn import multi_scale_deformable_attn_pytorch
from tt.utils import add_dim, get_list_shape

from bevformer.utils import assert_many

__all__ = ['CustomMSDeformableAttention']

class CustomMSDeformableAttention(BaseModule):
    count = 0
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = op.Linear(embed_dims, num_heads * num_levels * num_points * 2, dtype=ttnn.bfloat16)
        self.attention_weights = op.Linear(embed_dims,
                                           num_heads * num_levels * num_points, dtype=ttnn.bfloat16)
        self.value_proj = op.Linear(embed_dims, embed_dims, dtype=ttnn.bfloat16)
        self.output_proj = op.Linear(embed_dims, embed_dims, dtype=ttnn.bfloat16)
        self.adder = op.Add(deallocate_input=True)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        assert_many(query, f'CustomMSDeformableAttention.input_query.{CustomMSDeformableAttention.count}')
        assert_many(key, f'CustomMSDeformableAttention.input_key.{CustomMSDeformableAttention.count}')

        assert_many(value, f'CustomMSDeformableAttention.input_value.{CustomMSDeformableAttention.count}')
        assert_many(reference_points, f'CustomMSDeformableAttention.reference_points.{CustomMSDeformableAttention.count}')
        
        if identity is None:
            identity = query.squeeze(1)
        if query_pos is not None:
            query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
            query_pos = ttnn.to_memory_config(query_pos, ttnn.DRAM_MEMORY_CONFIG)
            query = query + query_pos
            # query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # assert (spatial_shapes.torch()[:, 0] * spatial_shapes.torch()[:, 1]).sum() == num_value, \
        #     f'num_value {num_value} {value.shape} is not equal to sum of spatial_shapes {spatial_shapes.torch()} = {(spatial_shapes.torch()[:, 0] * spatial_shapes.torch()[:, 1]).sum()}'

        value = self.value_proj(value)
        assert_many(value.squeeze(0), f'CustomMSDeformableAttention.value_projection.{CustomMSDeformableAttention.count}')
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        assert_many(value, f'CustomMSDeformableAttention.value_projection_reshaped.{CustomMSDeformableAttention.count}')        
        print(f"[CustomMSDeformableAttention] Reshape sampling offsets to {bs, num_query, self.num_heads, self.num_levels, self.num_points, 2}")
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        assert_many(sampling_offsets, f'CustomMSDeformableAttention.sampling_offsets.{CustomMSDeformableAttention.count}')
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        assert_many(attention_weights, f'CustomMSDeformableAttention.attention_weights.{CustomMSDeformableAttention.count}')
        attention_weights = attention_weights.softmax(-1)
        attention_weights = ttnn.to_memory_config(attention_weights, ttnn.DRAM_MEMORY_CONFIG)
        attention_weights = attention_weights.view(bs, 
                                                   num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        assert_many(attention_weights, f'CustomMSDeformableAttention.attention_weights_softmax.{CustomMSDeformableAttention.count}')
        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (reference_points.torch()[:, :, None, :, None, :] + sampling_offsets.torch() \
                                   / offset_normalizer.torch()[None, None, None, :, None, :]
                                  ) # .squeeze(0)
            #TODO: FALLBACK - div errors
            # sampling_locations = ttnn.from_torch(
            #     reference_points.torch()[:, :, None, :, None, :] + sampling_offsets.torch() / offset_normalizer.torch()[None, None, None, :, None, :], 
            #     device=self.device).squeeze(0)
            
        elif reference_points.shape[-1] == 4:
            #TODO: FALLBACK - div errors
            sampling_locations = reference_points.torch()[:, :, None, :, None, :2] \
                + sampling_offsets.torch() / self.num_points \
                * reference_points.torch()[:, :, None, :, None, 2:] \
                * 0.5 # .squeeze(0)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        output = ttnn.from_torch(multi_scale_deformable_attn_pytorch(ttnn.to_torch(value), 
                                                                     ttnn.to_torch(spatial_shapes, dtype=torch.int32), 
                                                                     sampling_locations, 
                                                                     ttnn.to_torch(attention_weights)),
                                 dtype=ttnn.bfloat16,
                                 device=self.device)
        assert_many(output, f'CustomMSDeformableAttention.output_kernel.{CustomMSDeformableAttention.count}')
        output = self.output_proj(output).squeeze(0, 0)
        # output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        # assert_many(output.unsqueeze(0), f'CustomMSDeformableAttention.output_projection.{CustomMSDeformableAttention.count}')
        if self.batch_first:
            output = output.row_major().permute(1, 0)

        CustomMSDeformableAttention.count += 1
        identity = ttnn.to_memory_config(identity, ttnn.L1_MEMORY_CONFIG)
        # output = ttnn.typecast(output, ttnn.bfloat16)
        output = output.tile() + identity
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return output.unsqueeze(1)