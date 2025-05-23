import ttnn
from bos_metal import op

class FFN(op.BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.,
                 add_identity=True,
                 **kwargs,
                 ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = op.Functional(ttnn.relu)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                op.Sequential(
                    op.Linear(in_channels, 
                              feedforward_channels,
                              dtype=ttnn.bfloat8_b,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG
                              ), 
                              self.activate,
                ))
            in_channels = feedforward_channels
        layers.append(op.Linear(feedforward_channels, embed_dims, dtype=ttnn.bfloat8_b))
        self.layers = op.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out