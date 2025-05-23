import math
import torch
import torch.nn as nn
from bos_metal import ttnn, op

__all__ = ['SinePositionalEncoding']

class SinePositionalEncoding(op.BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 **kwargs
                 ):
        super(SinePositionalEncoding, self).__init__(**kwargs)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.dim_t = ttnn.from_torch(torch.arange(self.num_feats, dtype=torch.bfloat16), device=self.device)

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        self.cout("Start SinePositionalEncoder")
        # `masks` from bool to int.
        not_mask = 1 + ttnn.multiply(mask.tile(), -1).tile()  # logical_not
        #TODO: FALLBACK cumsum
        not_mask_ = not_mask.torch()
        y_embed = not_mask_.cumsum(1, dtype=torch.float32)
        y_embed = ttnn.from_torch(y_embed, device=self.device, dtype=ttnn.bfloat16)
        x_embed = not_mask_.cumsum(2, dtype=torch.float32)
        x_embed = ttnn.from_torch(x_embed, device=self.device, dtype=ttnn.bfloat16)
        
        if self.normalize:
            y_embed = (y_embed.tile() + self.offset) / \
                      (y_embed[:, -1:, :].tile() + self.eps) * self.scale
            x_embed = (x_embed.tile() + self.offset) / \
                      (x_embed[:, :, -1:].tile() + self.eps) * self.scale

        exp_ = 2 * (self.dim_t // 2) / self.num_feats
        dim_t = self.temperature**exp_
        dim_t_repc = ttnn.reciprocal(dim_t)
        pos_x = dim_t_repc * x_embed.unsqueeze(-1).tile() 
        pos_y = dim_t_repc * y_embed.unsqueeze(-1).tile() 
        B, H, W = mask.size()
        
        pos_x = ttnn.stack(
            [ttnn.sin(pos_x[:, :, :, 0::2]), 
             ttnn.cos(pos_x[:, :, :, 1::2])],
            dim=0
        ).permute(1, 2, 3, 4, 0)    # stack last dim error for memory allocation
        pos_x = pos_x.view(B, H, W, -1)
        pos_y = ttnn.stack(
            (ttnn.sin(pos_y[:, :, :, 0::2]), 
             ttnn.cos(pos_y[:, :, :, 1::2])),
            dim=0
        ).permute(1, 2, 3, 4, 0)
        pos_y = pos_y.view(B, H, W, -1)
        pos = ttnn.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        self.cout("Finish SinePositionalEncoder")
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str