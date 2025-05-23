import torch
import ttnn
import ttnn.types
from bos_metal.core import BaseModule
from bos_metal import device_box, op, helpers
from tt.modules.blocks.attn.ms_deform_attn import multi_scale_deformable_attn_pytorch
from tt.utils import add_dim, get_list_shape

from bevformer.utils import assert_many

__all__ = ['TemporalSelfAttention']

class TemporalSelfAttention(BaseModule):
    count = 0
    """An attention module used in BEVFormer based on Deformable-Detr.

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
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 ):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False


        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.value_proj = op.Linear(embed_dims, embed_dims,
                                    dtype=ttnn.bfloat16,
                                    memory_config=ttnn.L1_MEMORY_CONFIG,
                                    )
        self.sampling_offsets = op.Linear(embed_dims*self.num_bev_queue, 
                                            num_bev_queue*num_heads * num_levels * num_points * 2,
                                            dtype=ttnn.bfloat16,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                            )
        self.attention_weights = op.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points,
                                           dtype=ttnn.bfloat16,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                           )
        self.output_proj = op.Linear(embed_dims, embed_dims)
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
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = ttnn.stack([query, query], 1).reshape([bs*2, len_bev, c])
        assert_many(query, f"TempAttn.query.{TemporalSelfAttention.count}")
        assert_many(query_pos, f"TempAttn.query_pos.{TemporalSelfAttention.count}")
        assert_many(value, f"TempAttn.value.{TemporalSelfAttention.count}")
        assert_many(reference_points, f"TempAttn.reference_points.{TemporalSelfAttention.count}")
        
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            assert_many(query, f"TempAttn.query_with_pos.{TemporalSelfAttention.count}")
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, [1, 0, 2])
            value = ttnn.permute(value, [1, 0, 2])
        
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        # assert (ttnn.to_torch(spatial_shapes)[:, 0] * ttnn.to_torch(spatial_shapes)[:, 1]).sum() == num_value
        # assert self.num_bev_queue == 2
        query = ttnn.concat([value[:bs].row_major(), query.row_major()], -1).tile()
        value = self.value_proj(value)
        # assert_many(value.squeeze(0), f"TempAttn.value_proj.{TemporalSelfAttention.count}")
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # value = ttnn.typecast(value, dtype=ttnn.bfloat16)
        value = value.reshape(bs*self.num_bev_queue,
                                num_value, self.num_heads, -1)
        value = value.torch()

        assert_many(query, f"TempAttn.query+for_offsets.{TemporalSelfAttention.count}")
        sampling_offsets = self.sampling_offsets(query)
        # assert_many(sampling_offsets.squeeze(0), f"TempAttn.MSDeform.sampling_offsets_from_query.{TemporalSelfAttention.count}")
        sampling_offsets = sampling_offsets.view(bs,
                                                 num_query,
                                                 self.num_heads,
                                                 self.num_bev_queue,
                                                 self.num_levels,
                                                 self.num_points,
                                                 2)
        assert_many(sampling_offsets, f"TempAttn.MSDeform.sampling_offsets_reshape.{TemporalSelfAttention.count}")
        attention_weights = self.attention_weights(query)
        # ttnn.deallocate(query)
        # assert_many(attention_weights.squeeze(0), f"TempAttn.MSDeform.attention_weights_from_query.{TemporalSelfAttention.count}")
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        assert_many(attention_weights, f"TempAttn.MSDeform.attention_weights_reshape.{TemporalSelfAttention.count}")   
        attention_weights = ttnn.softmax(attention_weights.tile(), -1)
        assert_many(attention_weights, f"TempAttn.MSDeform.attention_weights_softmax.{TemporalSelfAttention.count}")
        attention_weights = attention_weights.view(bs, 
                                                   num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape([
            bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points
        ])
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape([
            bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2
        ])
        assert_many(sampling_offsets, f"TempAttn.MSDeform.sampling_offsets.{TemporalSelfAttention.count}")
        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # assert_many(offset_normalizer.squeeze(0), f"TempAttn.MSDeform.offset_normalizer.{TemporalSelfAttention.count}")
            #TODO: FALLBACK - ttnn.multiply, ttnn.add give wrong outputs
            sampling_offsets_norm = sampling_offsets.torch() * \
                ttnn.reciprocal(offset_normalizer.tile()).unsqueeze(0, 0, 0, -2).torch()
            sampling_locations = reference_points.unsqueeze(2, 4).torch() + sampling_offsets_norm
            # assert_many(sampling_locations.squeeze(0), f"TempAttn.MSDeform.sampling_locations_norm.{TemporalSelfAttention.count}")
        elif reference_points.shape[-1] == 4:
            sampling_locations = (reference_points.unsqueeze(2, 5) + sampling_offsets.tile() / self.num_points \
                * reference_points.unsqueeze(2, -2)[..., 2:].tile() \
                * 0.5).torch()
        sampling_locations = sampling_locations.squeeze(0)  
        assert_many(sampling_locations, f"TempAttn.MSDeform.sampling_locations.{TemporalSelfAttention.count}")
        assert_many(attention_weights, f"TempAttn.MSDeform.attention_weights.{TemporalSelfAttention.count}")
        assert_many(value, f"TempAttn.MSDeform.value.{TemporalSelfAttention.count}")
        assert_many(spatial_shapes, f"TempAttn.MSDeform.spatial_shapes.{TemporalSelfAttention.count}")
        assert_many(level_start_index, f"TempAttn.MSDeform.level_start_index.{TemporalSelfAttention.count}")
        
        output = ttnn.from_torch(multi_scale_deformable_attn_pytorch(value, 
                                                                     ttnn.to_torch(spatial_shapes, dtype=torch.int32), 
                                                                     sampling_locations, 
                                                                     ttnn.to_torch(attention_weights)),
                                 dtype=ttnn.bfloat16,
                                 device=self.device)
        assert_many(output, f"TempAttn.MSDeform.{TemporalSelfAttention.count}")

        output = ttnn.permute(output, [1, 2, 0])

        output = ttnn.reshape(output, [num_query, embed_dims, bs, self.num_bev_queue])
        output = ttnn.mean(output, dim=-1)
        output = ttnn.reshape(output, get_list_shape(output)[:-1])
        
        output = ttnn.permute(output, [2, 0, 1])
        output = self.output_proj(output)
        # assert_many(output.squeeze(0), f"TempAttn.output_proj.{TemporalSelfAttention.count}")
        output = ttnn.reshape(output, get_list_shape(output)[1:])
        output - ttnn.to_layout(output, ttnn.TILE_LAYOUT, device=self.device)

        if not self.batch_first:
            output = ttnn.permute(output, [1, 0, 2])
        
        TemporalSelfAttention.count += 1
        return output + identity




if __name__ == "__main__":
    # Define parameters
    batch_size = 3
    num_query = 100  
    embed_dims = 256  
    num_heads = 8
    num_levels = 1 
    num_points = 4 
    num_bev_queue = 2  
    height, width = 20, 20  

    # Correct spatial shapes


    # Creating random tensors with the given shapes
    query = torch.randn(3, 2500, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    query_pos = torch.randn(3, 2500, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    reference_points = torch.rand(6, 2500, 1, 4, dtype=torch.bfloat16)  # (batch_size=6, num_query=2500, num_levels=1, xy=2 or 4)
    spatial_shapes = torch.tensor([[50, 50]])  # (num_levels=1, height & width)
    level_start_index = torch.tensor([0])  # (num_levels=1)
    
    # Run session
    device = device_box.get()
    # Instantiate the module
    temporal_attention = TemporalSelfAttention(
        embed_dims=embed_dims, 
        num_heads=num_heads, 
        num_levels=num_levels, 
        num_points=num_points, 
        num_bev_queue=num_bev_queue,
    )

    # Forward pass
    output = temporal_attention.forward(
        query=ttnn.from_torch(query, device=device),
        query_pos=ttnn.from_torch(query_pos, device=device),
        reference_points=ttnn.from_torch(reference_points, device=device),
        spatial_shapes=ttnn.from_torch(spatial_shapes, device=device),
        level_start_index=level_start_index,
        device=device
    )
    
    # Output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, num_query, embed_dims)
    device_box.close()