import torch
import torch.nn as nn
import ttnn
from bos_metal.core import BaseModule
import bos_metal.helpers as helpers
from bos_metal import device_box, op

class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """


    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 device=None,
                 **kwargs):
        super(FFN, self).__init__(device=device)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        # self.activate = ttnn.relu
        self.device = device
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                op.Sequential(
                    op.Linear(in_channels, feedforward_channels, requires_shape=False, activation='relu'),
                    nn.Identity(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(op.Linear(feedforward_channels, embed_dims, requires_shape=False))
        layers.append(nn.Identity(ffn_drop))
        self.layers = op.Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
    
    
    


if __name__ == '__main__':
    # For testing 
    embed_dims = 256  
    feedforward_channels = 512
    num_fcs = 2
    ffn_drop = 0.1
    act_cfg = dict(type='ReLU', inplace=True)
    inputs = torch.randn(3, 2500,256, dtype=torch.bfloat16)
    device = device_box.open({"device_id": 3})
    # Test FFN
    ffn = FFN(embed_dims, feedforward_channels, num_fcs, act_cfg, ffn_drop, device=device)
    outputs = ffn.forward(ttnn.from_torch(inputs, device=device))
    print("Output shape: ",outputs.shape)
    device_box.close()