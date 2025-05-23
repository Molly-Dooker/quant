import torch
import ttnn
import warnings
from bos_metal import device_box
from tt.modules.blocks.transformer.custom_base_transformer import MyCustomBaseTransformerLayer

from bevformer.utils import assert_many

__all__ = ['BEVFormerLayer']

class BEVFormerLayer(MyCustomBaseTransformerLayer):
    count = 0
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
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

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, ttnn.Tensor):
            attn_masks = [
                attn_masks.clone() for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        assert_many(query, f"bevformerlayer.in.{BEVFormerLayer.count}")
        order = 0
        for layer in self.operation_order:
            # temporal self attention
            while query.shape.rank >= 4:
                    query = ttnn.squeeze(query, 0)
            if layer == 'self_attn':
                query = self.attentions[attn_index].forward(
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=ttnn.from_torch(torch.Tensor([[bev_h, bev_w]]), dtype=ttnn.bfloat16, device=self.device, memory_config = ttnn.L1_MEMORY_CONFIG),
                    level_start_index=ttnn.from_torch(torch.Tensor([0]), dtype=ttnn.uint8, device=self.device, memory_config = ttnn.L1_MEMORY_CONFIG),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index].forward(query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index].forward(
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index].forward(
                    query, identity if self.pre_norm else None).squeeze(0)
                ffn_index += 1
            #
            assert_many(query, f"bevformerlayer.{layer}.{BEVFormerLayer.count}.{order}")
            order += 1
                
        BEVFormerLayer.count += 1
        return query
    
    
    


if __name__ == '__main__': 
    # Create input tensors
    attn_cfgs = [{'type': 'TemporalSelfAttention', 
                                'embed_dims': 256, 
                                'num_levels': 1}, 
                {'type': 'SpatialCrossAttention', 
                                'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                                'deformable_attention': {'embed_dims': 256, 
                                                        'num_points': 8, 
                                                        'num_levels': 1}, 
                                'embed_dims': 256}]
    feedforward_channels = 512
    ffn_dropout = 0.1
    operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')


    query = torch.randn(3, 2500, 256, dtype=torch.bfloat16)
    key = torch.randn(6, 375, 3, 256, dtype=torch.bfloat16)
    value = torch.randn(6, 375, 3, 256, dtype=torch.bfloat16)
    bev_pos = torch.randn(3, 2500, 256, dtype=torch.bfloat16)
    query_pos = None
    key_pos = None
    attn_masks = None
    query_key_padding_mask = None
    key_padding_mask = None
    ref_2d = torch.randn(6, 2500, 1, 2, dtype=torch.bfloat16)
    ref_3d = torch.randn(3, 4, 2500, 3, dtype=torch.bfloat16)
    bev_h = 50
    bev_w = 50
    reference_points_cam = torch.randn(6, 3, 2500, 4, 2, dtype=torch.bfloat16)
    mask = None
    bev_mask = torch.randint(0, 2, (6, 3, 2500, 4), dtype=torch.bfloat16)
    spatial_shapes = torch.tensor([[15, 25]])
    level_start_index = torch.Tensor([0])
    prev_bev = None
    # Init device
    device = device_box.open({"device_id": 1})
    # Inint class
    layer = BEVFormerLayer(attn_cfgs=attn_cfgs,
                           feedforward_channels=feedforward_channels,
                           ffn_dropout=ffn_dropout,
                           operation_order=operation_order,
                           device=device)
    
    output = layer.forward(query=ttnn.from_torch(query, 
                                                 memory_config=ttnn.L1_MEMORY_CONFIG,
                                                 device=device),
                            key=ttnn.from_torch(key, 
                                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                                device=device),
                            value=ttnn.from_torch(value, 
                                                  memory_config=ttnn.L1_MEMORY_CONFIG,
                                                  device=device),
                            bev_pos=ttnn.from_torch(bev_pos, 
                                                    memory_config=ttnn.L1_MEMORY_CONFIG,
                                                    device=device),
                            query_pos=query_pos,
                            key_pos=key_pos,
                            attn_masks=attn_masks,
                            query_key_padding_mask=query_key_padding_mask,
                            key_padding_mask=key_padding_mask,
                            ref_2d=ttnn.from_torch(ref_2d, 
                                                   memory_config=ttnn.L1_MEMORY_CONFIG,
                                                   device=device),
                            ref_3d=ttnn.from_torch(ref_3d, 
                                                   memory_config=ttnn.L1_MEMORY_CONFIG,
                                                   device=device),
                            bev_h=bev_h,
                            bev_w=bev_w,
                            reference_points_cam=ttnn.from_torch(reference_points_cam, 
                                                                 memory_config=ttnn.L1_MEMORY_CONFIG,
                                                                 device=device),
                            bev_mask=ttnn.from_torch(bev_mask, 
                                                     memory_config=ttnn.L1_MEMORY_CONFIG,
                                                     device=device),
                            mask=mask,
                            spatial_shapes=ttnn.from_torch(spatial_shapes, 
                                                           memory_config=ttnn.L1_MEMORY_CONFIG,
                                                           dtype=ttnn.bfloat16,
                                                           device=device),
                            level_start_index=level_start_index,
                            prev_bev=prev_bev)
    
    print("Output shape: ",output.shape)
    device_box.close()
    
    

 
                   
    