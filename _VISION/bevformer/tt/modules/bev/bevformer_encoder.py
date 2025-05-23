import torch
import ttnn
import numpy as np
from tt.modules.blocks.transformer.transformer_layer_sequence import TransformerLayerSequence
from tt.utils import from_torch_many
from bevformer.utils import assert_many

__all__ = ['BEVFormerEncoder']

class BEVFormerEncoder(TransformerLayerSequence):
    count = 0

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4,
                 bev_h=None,
                 bev_w=None,
                 return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bs = 1
    
        self._init_tensors()

    def _init_tensors(self):
        def pt2tt(tensor, dtype=ttnn.bfloat16):
            return ttnn.from_torch(tensor, dtype=dtype)

        ref_3d = self.get_reference_points(
            self.bev_h, 
            self.bev_w, 
            self.pc_range[5]-self.pc_range[2], 
            self.num_points_in_pillar, 
            dim='3d', 
            bs=self.bs, 
            device=self.device, 
        )
        ref_2d = self.get_reference_points(
            self.bev_h, 
            self.bev_w, 
            dim='2d', 
            bs=self.bs, 
            device=self.device, 
        )

        self.ref_3d = pt2tt(ref_3d)
        self.ref_2d = pt2tt(ref_2d)
        # assert_many(self.ref_3d, f"bevformerencoder.ref_3d")
        # assert_many(self.ref_2d, f"bevformerencoder.ref_2d")

    @staticmethod
    #NOTE: this is used as re-inititation
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device=None):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        def linspace3d(length, size) -> torch.Tensor:   #type: ignore
            nonlocal device
            tensor_ = torch.linspace(0.5, length - 0.5, size, device='cpu', dtype=torch.float32)
            return tensor_

        def linspace2d(length) -> torch.Tensor:  #type: ignore
            nonlocal device
            tensor_ = torch.linspace(0.5, length - 0.5, length, device='cpu')
            return tensor_

        if dim == '3d':
            zs = linspace3d(Z, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = linspace3d(W, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = linspace3d(H, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(linspace2d(H), linspace2d(W))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @staticmethod
    def get_reference_points_(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device=None, dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        #TODO: FALLBACK
        def linspace3d(length, size) -> ttnn.Tensor:   #type: ignore
            nonlocal device
            tensor_ = torch.linspace(0.5, length - 0.5, size, device='cpu', dtype=torch.float32)
            return ttnn.from_torch(tensor_, dtype=ttnn.float32, device=device)
        
        #TODO: FALLBACK
        def linspace2d(length) -> ttnn.Tensor:  #type: ignore
            nonlocal device
            tensor_ = torch.linspace(0.5, length - 0.5, length, device='cpu')
            return tensor_
        
        if dim == '3d':
            zs = linspace3d(Z, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W).tile() / Z
            xs = linspace3d(W, W).view(1, 1, W).expand(num_points_in_pillar, H, W).tile() / W
            ys = linspace3d(H, H).view(1, H, 1).expand(num_points_in_pillar, H, W).tile() / H
            ref_3d = ttnn.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ttnn.partition_repeat(ref_3d.unsqueeze(0), bs, 0)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = from_torch_many(torch.meshgrid(linspace2d(H),linspace2d(W)), 
                                            device=device)
            ref_y = ref_y.unsqueeze(0).tile() / H
            ref_y = ref_y.reshape([1, H*W])
            ref_x = ref_x.unsqueeze(0).tile() / W
            ref_x = ref_x.reshape([1, H*W])            
            ref_2d = ttnn.stack((ref_x, ref_y), -1)
            ref_2d = ttnn.partition_repeat(ref_2d, bs, 0).unsqueeze(2)
            return ref_2d

    # def point_sampling(self, reference_points, pc_range,  img_metas):
    #     # NOTE: close tf32 here.
    #     lidar2img = []
    #     for img_meta in img_metas:
    #         lidar2img.append(img_meta['lidar2img'])
    #     lidar2img = np.asarray(lidar2img)
    #     assert_many(lidar2img, f"bevformerencoder.{BEVFormerEncoder.count}.lidar2img")
    #     lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    #     reference_points = reference_points.clone()
    #     assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_clone")

    #     reference_points[..., 0:1] = reference_points[..., 0:1] * \
    #         (pc_range[3] - pc_range[0]) + pc_range[0]
    #     assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem01")
    #     reference_points[..., 1:2] = reference_points[..., 1:2] * \
    #         (pc_range[4] - pc_range[1]) + pc_range[1]
    #     assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem12")
    #     reference_points[..., 2:3] = reference_points[..., 2:3] * \
    #         (pc_range[5] - pc_range[2]) + pc_range[2]
    #     assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem23")
    #     reference_points = torch.cat(
    #         (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    #     reference_points = reference_points.permute(1, 0, 2, 3)
    #     D, B, num_query = reference_points.size()[:3]
    #     num_cam = lidar2img.size(1)
    #     reference_points = reference_points.view(
    #         D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    #     lidar2img = lidar2img.view(
    #         1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    #     reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
    #                                         reference_points.to(torch.float32)).squeeze(-1)
    #     assert_many(reference_points_cam, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_cam_matmul")
    #     eps = 1e-5

    #     bev_mask = (reference_points_cam[..., 2:3] > eps)
    #     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    #     reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    #     reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    #     bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
    #                 & (reference_points_cam[..., 1:2] < 1.0)
    #                 & (reference_points_cam[..., 0:1] < 1.0)
    #                 & (reference_points_cam[..., 0:1] > 0.0))
    #     bev_mask = torch.nan_to_num(bev_mask)
    #     reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    #     bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    #     # reference_points_cam = reference_points_cam.to(torch.bfloat16)
    #     # bev_mask = bev_mask.to(torch.bfloat16)
    #     # reference_points_cam = ttnn.from_torch(reference_points_cam, dtype=ttnn.bfloat16, device=self.device)
    #     # bev_mask = ttnn.from_torch(bev_mask, dtype=ttnn.bfloat16, device=self.device)
    #     return reference_points_cam, bev_mask
    
    #TODO: FALLBACK 
    # This function must use fp32!!!
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = ttnn.stack(lidar2img[0], dim=0).unsqueeze(0) # (B, N, 4, 4)
        # assert_many(lidar2img, f"bevformerencoder.{BEVFormerEncoder.count}.lidar2img")
        
        reference_points = reference_points.clone().tile()
        reference_points = reference_points.permute(3, 0, 1, 2)
        # assert_many(reference_points.permute(1, 2, 3, 0), f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_clone")
        #TODO: FALLBACK small slice start > 1
        # reference_points = reference_points.torch()
        ref_x = reference_points[0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # reference_points[..., 0:1] = ref_x
        # assert_many(reference_points.permute(1, 2, 3, 0), f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem01")
        ref_y = reference_points[1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_points[..., 1:2] = ref_y
        # assert_many(reference_points.permute(1, 2, 3, 0), f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem12")
        ref_z = reference_points[2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        # reference_points[..., 2:3] = ref_z
        reference_points = ttnn.concat((ref_x, ref_y, ref_z), 0).permute(1, 2, 3, 0)
        # assert_many(reference_points.permute(1, 2, 3, 0), f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_setitem23")

        # reference_points = from_torch_many(reference_points)        
        reference_points = ttnn.concat((
            reference_points.row_major(), 
            ttnn.ones_like(reference_points[..., :1]).row_major()
        ), -1).tile()
        assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_concat")
        reference_points = reference_points.permute(1, 0, 2, 3)
        assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_permute")
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4)
        reference_points = ttnn.partition_repeat(reference_points, num_cam, 2)
        reference_points = reference_points.unsqueeze(-1).tile()
        assert_many(reference_points, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_repeats")
        
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4)
        lidar2img = ttnn.partition_repeat(lidar2img, num_query, 3)
        lidar2img = ttnn.partition_repeat(lidar2img, D, 0).tile()
        assert_many(lidar2img, f"bevformerencoder.{BEVFormerEncoder.count}.lidar2img_repeats")

        reference_points_cam = ttnn.matmul(lidar2img, reference_points).squeeze(-1)
        assert_many(reference_points_cam, f"bevformerencoder.{BEVFormerEncoder.count}.reference_points_cam_matmul")

        eps = 1e-5
        reference_points_cam = ttnn.permute(reference_points_cam, (4,0,1,2,3))

        bev_mask = (reference_points_cam[2:3] > eps) #.torch(dtype=torch.bool)
        reference_points_cam = ttnn.divide(reference_points_cam[0:2], ttnn.repeat_interleave(ttnn.maximum(
            reference_points_cam[2:3], ttnn.full_like(reference_points_cam[2:3], eps)), 2, 0)) 
        reference_points_cam_0 = reference_points_cam[0:1] / img_metas[0]['img_shape'][0][1]
        reference_points_cam_1 = reference_points_cam[1:2] / img_metas[0]['img_shape'][0][0]
        reference_points_cam = ttnn.concat((reference_points_cam_0, reference_points_cam_1), 0)
        # reference_points_cam = reference_points_cam.permute(1, 2, 3, 4, 0)

        bev_mask = ttnn.logical_and(ttnn.gt(reference_points_cam[1:2], 0.), bev_mask)
        bev_mask = ttnn.logical_and(ttnn.lt(reference_points_cam[1:2], 1.), bev_mask)
        bev_mask = ttnn.logical_and(ttnn.gt(reference_points_cam[0:1], 0.), bev_mask)
        bev_mask = ttnn.logical_and(ttnn.lt(reference_points_cam[0:1], 1.), bev_mask)

        # nan_idx = ttnn.isnan(bev_mask)
        # if not nan_idx.squeeze(-1).max() == 0 :
        #     bev_mask = ttnn.where(nan_idx, 0, bev_mask)
        
        #TODO: FALLBACK
        reference_points_cam = reference_points_cam.permute(3, 2, 4, 1, 0).torch()
        bev_mask = bev_mask.permute(3, 2, 4, 1, 0).squeeze(-1).torch(dtype=torch.bool)

        return reference_points_cam, bev_mask

    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        
        intermediate = []
        self.ref_3d = self.ref_3d.to(self.device)
        self.ref_2d = self.ref_2d.to(self.device)
        ref_3d = self.ref_3d
        assert_many(ref_3d, f"bevformerencoder.ref_3d.{BEVFormerEncoder.count}")
        ref_2d = self.ref_2d
        assert_many(ref_2d, f"bevformerencoder.ref_2d.{BEVFormerEncoder.count}")
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])
        assert_many(reference_points_cam, f"bevformerencoder.reference_points_cam.{BEVFormerEncoder.count}")
        assert_many(bev_mask, f"bevformerencoder.bev_mask.{BEVFormerEncoder.count}")
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        ref_2d = ref_2d.tile()
        shift_ref_2d = ttnn.add_(ref_2d, shift.unsqueeze(1, 2).tile()).row_major()
        # shift_ref_2d = ref_2d.clone()
        # shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.size()
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = ttnn.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = ttnn.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = ttnn.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
        
        order = 0
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            assert_many(output, f"bevformerencoder.layer.{BEVFormerEncoder.count}.{order}")
            order += 1
            if self.return_intermediate:
                intermediate.append(output)
                
        if self.return_intermediate:
            output = ttnn.stack(intermediate)

        return output
