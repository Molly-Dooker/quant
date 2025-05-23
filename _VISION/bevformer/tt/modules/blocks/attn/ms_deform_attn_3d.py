import warnings
import warnings
import torch
import ttnn
import ttnn.types
from bos_metal.core import BaseModule
from bos_metal import device_box, op
from .ms_deform_attn import multi_scale_deformable_attn_pytorch
from tt.utils import add_dim

from bevformer.utils import assert_many, load_many

__all__ = ['MSDeformableAttention3D']

class MSDeformableAttention3D(BaseModule):
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
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = op.Linear(
            embed_dims, num_heads * num_levels * num_points * 2, 

            )
        self.attention_weights = op.Linear(embed_dims,
                                           num_heads * num_levels * num_points,
                                           dtype=ttnn.bfloat16
                                           )
        self.value_proj = op.Linear(embed_dims, embed_dims, 
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, 
                                    dtype=ttnn.bfloat16
                                    )
        self.add = op.Add(deallocate_input=True)
        self.add = op.Add(deallocate_input=True)

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
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
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
        if identity is None:
            identity = query
        # 
        # query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        # key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)    
        # value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)   
        # assert_many(query.squeeze(0), f"ms3d.query.{MSDeformableAttention3D.count}") 
        # assert_many(key.squeeze(0), f"ms3d.key.{MSDeformableAttention3D.count}") 
        # assert_many(value.squeeze(0), f"ms3d.value.{MSDeformableAttention3D.count}") 
        assert_many(query_pos, f"ms3d.query_pos.{MSDeformableAttention3D.count}") 
        assert_many(reference_points, f"ms3d.reference_points.{MSDeformableAttention3D.count}")
                
        if query_pos is not None:
            query = query + query_pos
            # assert_many(query.squeeze(0), f"ms3d.query_pos_add.{MSDeformableAttention3D.count}") 

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
        # assert_many(query.squeeze(0), f"ms3d.query_reshape.{MSDeformableAttention3D.count}") 
        # assert_many(value.squeeze(0), f"ms3d.value_reshape.{MSDeformableAttention3D.count}") 
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        
        # assert (spatial_shapes.torch()[:, 0] * spatial_shapes.torch()[:, 1]).sum() == num_value, \
            # f'num_value {num_value} is not equal to sum of spatial_shapes {spatial_shapes.torch()} = {(spatial_shapes.torch()[:, 0] * spatial_shapes.torch()[:, 1]).sum()}'

        value = self.value_proj(value)
        # value = ttnn.typecast(value, ttnn.bfloat16)
        # value = ttnn.clone(value, dtype=ttnn.bfloat16)
        # assert_many(value.squeeze(0), f"ms3d.value_proj.{MSDeformableAttention3D.count}")
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        # assert_many(value.squeeze(0), f"ms3d.value_view.{MSDeformableAttention3D.count}")
        sampling_offsets = self.sampling_offsets(query)
        #TODO: FALLBACK
        sampling_offsets_ = sampling_offsets.torch()
        ttnn.deallocate(sampling_offsets)
        sampling_offsets = sampling_offsets_.reshape(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        assert_many(sampling_offsets, f"ms3d.sampling_offsets_query_reshape.{MSDeformableAttention3D.count}")
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.reshape(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        attention_weights_ = attention_weights.reshape([bs, num_query, self.num_heads, self.num_levels, self.num_points])
        attention_weights = attention_weights_.torch()
        ttnn.deallocate(attention_weights_)
        assert_many(attention_weights, f"ms3d.attention_weights.{MSDeformableAttention3D.count}")
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = ttnn.concat(
                [spatial_shapes[..., 1:2], spatial_shapes[..., 0:1]], 
                -1,
                ).unsqueeze(0, 0, 0, -2).tile()
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            #TODO: FALLBACK - div errors: Index is out of bounds for the rank, should be between 0 and 5 however is 6
            sampling_offsets = sampling_offsets / offset_normalizer.torch()
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            # `num_query` last for ttnn.add allocation efficiency
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
            )
            # Add offset to reference points
            #TODO: FALLBACK - add not correct
            sampling_locations = reference_points.unsqueeze(2, 3, 4).torch() + sampling_offsets
            # Get the shape of sampling locations
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            
            sampling_locations = sampling_locations.reshape([
                bs, num_query, num_heads, num_levels, num_all_points, xy
            ])      

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        assert_many(sampling_locations, f"ms3d.sampling_locations.{MSDeformableAttention3D.count}")
        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        #TODO: FALLBACK
        assert_many(value, f"ms3d.value_last.{MSDeformableAttention3D.count}")
        assert_many(spatial_shapes, f"ms3d.spatial_shapes.{MSDeformableAttention3D.count}")

        output = ttnn.from_torch(multi_scale_deformable_attn_pytorch(ttnn.to_torch(value, dtype=torch.float32), 
                                                                      ttnn.to_torch(spatial_shapes, dtype=torch.int32), 
                                                                      sampling_locations, 
                                                                      attention_weights),
                                dtype = ttnn.bfloat16,
                                device=self.device)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        assert_many(output, f"ms3d.output.{MSDeformableAttention3D.count}")
        MSDeformableAttention3D.count += 1

        return output




if __name__ == "__main__":
    # Define parameters
    batch_size = 2
    num_query = 100  
    embed_dims = 256  
    num_heads = 8
    im2col_step = 64
    num_levels = 1 
    num_points = 8

    num_bev_queue = 2  
    height, width = 20, 20  

    # Correct spatial shapes

    # Creating random tensors with the given shapes
    query = torch.randn(18, 1183, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    key = torch.randn(18, 375, 256, dtype=torch.bfloat16)  # (batch_size=3, num_key=2500, embed_dims=256)
    value = torch.randn(18, 375, 256, dtype=torch.bfloat16)  # (batch_size=3, num_key=2500, embed_dims=256)

    # query_pos = torch.randn(3, 2500, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    reference_points = torch.rand(18, 1183, 4, 2, dtype=torch.bfloat16)  # (batch_size=6, num_query=2500, num_levels=1, xy=2)
    spatial_shapes = torch.tensor([[15, 25]])  # (num_levels=1, height & width)
    level_start_index = torch.tensor([0])  # (num_levels=1)
    # Compute level_start_index correctly

    device = device_box.get()
    # Instantiate the module
    msdeformattn3d = MSDeformableAttention3D(
        embed_dims=embed_dims, 
        num_heads=num_heads, 
        num_levels=num_levels, 
        num_points=num_points, 
        im2col_step=im2col_step    
    )

    # Forward pass
    output = msdeformattn3d.forward(
        query=ttnn.from_torch(query, device=device),
        key=ttnn.from_torch(key, device=device),
        value=ttnn.from_torch(value, device=device),
        reference_points=ttnn.from_torch(reference_points, device=device),
        spatial_shapes=ttnn.from_torch(spatial_shapes, device=device),
        level_start_index=level_start_index,
        device=device
    )
    
    # Output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, num_query, embed_dims)
    device_box.close()