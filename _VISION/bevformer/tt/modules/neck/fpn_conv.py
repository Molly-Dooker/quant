from typing import Tuple, Union
from bos_metal import op
from tt.modules.ops import Conv2dSplit

__all__ = [
    'FPNConvModule'
]

class FPNConvModule(op.BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 act='',
                 bias: Union[bool, str] = 'auto',
                 padding_mode: str = 'zeros',
                 ):
        super().__init__()
        official_padding_mode = ['zeros', 'circular']
        self.with_explicit_padding = padding_mode not in official_padding_mode

        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = True #not self.with_norm

        if self.with_explicit_padding:
            self.padding = padding

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = Conv2dSplit(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            activation=act
        )

    def forward(self, x):
        x = self.conv(x, threshold=2227200)
        return x
 