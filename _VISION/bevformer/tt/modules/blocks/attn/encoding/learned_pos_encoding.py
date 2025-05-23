import ttnn
from bos_metal import op

from bevformer.utils import assert_many

__all__ = ['LearnedPositionalEncoding']

class LearnedPositionalEncoding(op.Operation):
    """Position embedding with learnable embedding weights. Mimicked from MMDet."""
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 *,
                 device=None,
                 init_cfg=None,
                 **kwargs):
        super(LearnedPositionalEncoding, self).__init__(device=device, init_config=init_cfg, **kwargs)
        # print("Initialize LearnedPositionalEncoding, device:", self.device)
        
        self.row_embed = op.Embedding(row_num_embed, num_feats, device=device)
        self.col_embed = op.Embedding(col_num_embed, num_feats, device=device)

        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # assert_many(mask.torch(), "learned_pos_enc.mask")
        h, w = mask.shape[-2], mask.shape[-1]
        x = ttnn.arange(0, w, device=self.device)
        y = ttnn.arange(0, h, device=self.device)
        # assert_many(x.torch().squeeze(0).squeeze(0).squeeze(0), "learned_pos_enc.x")
        # assert_many(y.torch().squeeze(0).squeeze(0).squeeze(0), "learned_pos_enc.y")

        x_embed = self.col_embed(x) # 1, w, num_feats
        y_embed = self.row_embed(y) # 1, h, num_feats
        # assert_many(x_embed.torch().squeeze(0), "learned_pos_enc.x_embed")
        # assert_many(y_embed.torch().squeeze(0), "learned_pos_enc.y_embed")
        y_embed = y_embed.permute(1, 0, 2)

        stack_x_embed = ttnn.partition_repeat(x_embed, h, 0)
        # stack_x_embed = ttnn.repeat_interleave(stack_x_embed, 2, 0)
        
        stack_y_embed = ttnn.partition_repeat(y_embed, w, 1)
        # stack_y_embed = ttnn.repeat_interleave(stack_y_embed, 2, 1)

        pos = ttnn.concat([
            stack_x_embed,
            stack_y_embed
        ],
            dim=-1
        )   # h, w, num_feats

        pos = ttnn.permute(pos, (2, 0, 1))
        pos = ttnn.repeat_interleave(ttnn.unsqueeze(pos, 0), mask.shape[0], 0)
        # assert_many(pos.torch(), "learned_pos_enc.pos")       

        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


if __name__=="__main__":
    import torch.nn as nn
    import torch
    from bos_metal import helpers, device_box

    config = op.EmbeddingConfig(dtype=ttnn.bfloat16)
    to_embed = nn.Embedding(50, 256)
    tt_embed = op.Embedding(50, 256, config=config)
    input = torch.Tensor([1, 2, 3, 4, 5]).long()
    input_tt = ttnn.from_torch(input, dtype=ttnn.uint32, device=device_box.get())
    tt_embed.load_state_dict(to_embed.state_dict())
    to_embed.eval()
    tt_embed.eval()
    with torch.no_grad():
        out = to_embed(input)
        out_tt = tt_embed(input_tt.row_major()) 
        #!BUG: TILE_LAYOUT wrong outputs
    
    # helpers.compare_tensors(out, out_tt.torch().squeeze(0))