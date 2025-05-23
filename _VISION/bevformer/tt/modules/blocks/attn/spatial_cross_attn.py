import torch
import ttnn
from bos_metal.core import BaseModule
from bos_metal import device_box, op
from tt.modules.blocks.attn.ms_deform_attn_3d import MSDeformableAttention3D
from tt.utils import get_list_shape
from bevformer.utils import assert_many

__all___ = ['SpatialCrossAttention']

class SpatialCrossAttention(BaseModule):
    count = 0
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                    #  type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__()

        self.init_cfg = init_cfg
        # self.dropout = torch.nn.Identity() # nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = MSDeformableAttention3D(**deformable_attention) # build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = op.Linear(embed_dims, embed_dims, requires_shape=False)
        self.batch_first = batch_first
        # self.device = device


    def forward(self,
                    query,
                    key,
                    value,
                    residual=None,
                    query_pos=None,
                    key_padding_mask=None,
                    reference_points=None,
                    spatial_shapes=None,
                    reference_points_cam=None,
                    bev_mask=None,
                    level_start_index=None,
                    flag='encoder',
                    **kwargs):
            """Forward Function of Detr3DCrossAtten.
            Args:
                query (Tensor): Query of Transformer with shape
                    (num_query, bs, embed_dims).
                key (Tensor): The key tensor with shape
                    `(num_key, bs, embed_dims)`.
                value (Tensor): The value tensor with shape
                    `(num_key, bs, embed_dims)`. (B, N, C, H, W)
                residual (Tensor): The tensor used for addition, with the
                    same shape as `x`. Default None. If None, `x` will be used.
                query_pos (Tensor): The positional encoding for `query`.
                    Default: None.
                key_pos (Tensor): The positional encoding for  `key`. Default
                    None.
                reference_points (Tensor):  The normalized reference
                    points with shape (bs, num_query, 4),
                    all elements is range in [0, 1], top-left (0,0),
                    bottom-right (1, 1), including padding area.
                    or (N, Length_{query}, num_levels, 4), add
                    additional two dimensions is (w, h) to
                    form reference boxes.
                key_padding_mask (Tensor): ByteTensor for `query`, with
                    shape [bs, num_key].
                spatial_shapes (Tensor): Spatial shape of features in
                    different level. With shape  (num_levels, 2),
                    last dimension represent (h, w).
                level_start_index (Tensor): The start index of each level.
                    A tensor has shape (num_levels) and can be represented
                    as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            Returns:
                Tensor: forwarded results with shape [num_query, bs, embed_dims].
            """
            ###
            
            if key is None:
                key = query
            if value is None:
                value = key

            if residual is None:
                inp_residual = query
                slots = ttnn.zeros_like(query)
            if query_pos is not None:
                query = query + query_pos
                assert_many(query, f'spatial_cross_attention.query_pos.{SpatialCrossAttention.count}')

            assert_many(query, f'spatial_cross_attention.query.{SpatialCrossAttention.count}')
            assert_many(reference_points_cam, f'spatial_cross_attention.reference_points_cam.{SpatialCrossAttention.count}')
            query_shape = query.shape
            bs = query_shape[0]
            num_query = query_shape[1]

            D = reference_points_cam.shape[3]
            indexes = []
            # bev_mask_ = [bev_mask[i:i+1, :, :, :] for i in range(bev_mask.shape[0])]
            # for i, mask_per_img in enumerate(bev_mask_):
            #     # index_query_per_img = mask_per_img[0].squeeze(0,0).sum(-1).torch().squeeze(-1).nonzero().squeeze(-1)
            #     index_query_per_img = ttnn.to_torch(mask_per_img[0].squeeze(0,0).sum(-1).row_major().squeeze(-1), dtype=torch.int32).nonzero().squeeze(-1)
            #     indexes.append(index_query_per_img)
            # max_len = max([len(each) for each in indexes])
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            max_len = max([len(each) for each in indexes])
            assert_many(indexes, f'spatial_cross_attention.indexes.{SpatialCrossAttention.count}')
            # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
            
            queries_rebatch = torch.zeros(size=[bs, self.num_cams, max_len, self.embed_dims], dtype=torch.bfloat16)
            reference_points_rebatch = torch.zeros(size=[bs, self.num_cams, max_len, D, 2], dtype=torch.bfloat16)
            assert_many(queries_rebatch, f'spatial_cross_attention.queries_rebatch.{SpatialCrossAttention.count}')
            # reference_points_cam = ttnn.to_torch(reference_points_cam)
            
            #TODO: FALLBACK: Not supported slicing with assignment
            query_ = query.row_major().torch()
            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):   
                    index_query_per_img = indexes[i]
                    queries_rebatch[j, i, :len(index_query_per_img)] = query_[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

            # assert_many(key.squeeze(2), f'spatial_cross_attention.key.{SpatialCrossAttention.count}')
            # assert_many(value.squeeze(2), f'spatial_cross_attention.value.{SpatialCrossAttention.count}')
            
            queries_rebatch = ttnn.from_torch(queries_rebatch, device=self.device)
            assert_many(queries_rebatch, f'spatial_cross_attention.queries_rebatch_add.{SpatialCrossAttention.count}')
            assert_many(reference_points_rebatch, f'spatial_cross_attention.reference_points_rebatch.{SpatialCrossAttention.count}')
            reference_points_rebatch = ttnn.from_torch(reference_points_rebatch, device=self.device)
            
            num_cams, l, bs, embed_dims = key.shape
            key = ttnn.reshape(ttnn.permute(key,(2, 0, 1, 3)),(
                bs * self.num_cams, l, self.embed_dims))
            value = ttnn.reshape(ttnn.permute(value,(2, 0, 1, 3)),(
                bs * self.num_cams, l, self.embed_dims))
            assert_many(key, f'spatial_cross_attention.key.{SpatialCrossAttention.count}')
            assert_many(value, f'spatial_cross_attention.value.{SpatialCrossAttention.count}')
            queries = self.deformable_attention.forward(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), 
                                                        key=key, 
                                                        value=value,
                                                        reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), 
                                                        spatial_shapes=spatial_shapes,
                                                        level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
            assert_many(queries, f'spatial_cross_attention.deformable_attention.queries.{SpatialCrossAttention.count}')
            slots_ = ttnn.to_torch(slots)
            queries_ = ttnn.to_torch(queries)
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    slots_[j, index_query_per_img] += queries_[j, i, :len(index_query_per_img)]
                    
            assert_many(slots_, f"spatial_cross_attention.slots_setitem.{SpatialCrossAttention.count}")
            slots = ttnn.from_torch(slots_, layout=ttnn.TILE_LAYOUT, device=self.device)
            bev_mask = ttnn.from_torch(bev_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            count = ttnn.sign(ttnn.reshape(ttnn.sum(bev_mask, -1), get_list_shape(bev_mask)[0:-1])) 
            count = ttnn.permute(count, (1, 2, 0))
            count = ttnn.reshape(ttnn.sum(count, -1), get_list_shape(count)[0:-1])
            count = ttnn.clamp(count, min=1.0)
            count = ttnn.to_layout(ttnn.reshape(count, get_list_shape(count) + [1]), ttnn.TILE_LAYOUT, device=self.device)
            slots = ttnn.div(slots,count)
            slots = self.output_proj(slots)
            # assert_many(slots.squeeze(0), f'spatial_cross_attention.slots.{SpatialCrossAttention.count}')
            SpatialCrossAttention.count += 1
            return (slots + inp_residual).squeeze(0)





# Key padding mask (optional)



if __name__ == "__main__":
    # Define parameters

    ## init 
    embed_dims = 256  
    num_cams = 6
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    batch_first = True
    deformable_attention = {'embed_dims': 256, 'num_points': 8, 'num_levels': 1} # 'type': 'MSDeformableAttention3D', 

    ## forward 
    batch_size = 2
    num_query = 100  
    num_heads = 8
    num_levels = 1 
    num_points = 4 
    num_bev_queue = 2  
    height, width = 20, 20  

    # Creating random tensors with the given shapes
    query = torch.randn(3, 2500, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    key = torch.randn(6, 375, 3, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)
    value = torch.randn(6, 375, 3, 256, dtype=torch.bfloat16)  # (batch_size=3, num_query=2500, embed_dims=256)

    reference_points = torch.rand(3, 4, 2500, 3, dtype=torch.bfloat16)  # (batch_size=6, num_query=2500, num_levels=1, xy=2)
    spatial_shapes = torch.tensor([[15, 25]])  # (num_levels=1, height & width)
    reference_points_cam = torch.rand(6, 3, 2500, 4, 2, dtype=torch.bfloat16)  # (batch_size=6, num_query=2500, num_levels=1, xy=2)
    bev_mask = torch.randint(0, 2, (6, 1, 2500, 4), dtype=torch.bfloat16)  # (batch_size=6, num_query=2500, num_levels=1)
    level_start_index = torch.tensor([0])  # (num_levels=1)
    
    ## Run session
    device = device_box.get()
    # Instantiate the module
    spatial_cross_attn = SpatialCrossAttention(
        embed_dims=embed_dims, 
        num_cams=num_cams, 
        pc_range=pc_range, 
        deformable_attention=deformable_attention, 
        batch_first=batch_first,
        device=device
    )

    # Forward pass
    output = spatial_cross_attn.forward(
        query=ttnn.from_torch(query, device=device),
        key=ttnn.from_torch(key, device=device),
        value=ttnn.from_torch(value, device=device),
        reference_points=ttnn.from_torch(reference_points, device=device),
        reference_points_cam=ttnn.from_torch(reference_points_cam, device=device),
        bev_mask=ttnn.from_torch(bev_mask, device=device),
        spatial_shapes=ttnn.from_torch(spatial_shapes, device=device),
        level_start_index=level_start_index,
        device=device
    )
    
    # Output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, num_query, embed_dims)
    device_box.close()
    
    