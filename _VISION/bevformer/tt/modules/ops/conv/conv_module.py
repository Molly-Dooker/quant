import ttnn
from bos_metal import op
from .split_conv import Conv2dSplit, Conv2dSplitChannels

class ConvModule(op.BaseModule):
    """Convolutional block with Batch normalization and ReLU activation."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 act="relu",
                 split_channels=False,
                 deallocate_input=True,
                 shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                 preprocess_state_dict=False,
                 bs_as_group = False,
                 batch_size = 1,
                 **kwargs
                ):
        super(ConvModule, self).__init__(**kwargs)

        #already fused batchnorm into conv2d
        self.bs_as_group = bs_as_group
        self.bs = batch_size
        self.preprocess_state_dict = preprocess_state_dict
        self.split_channels = split_channels
        self.in_channels = in_channels
        # Compute sub_in, sub_out for split channel
        if split_channels:
            if in_channels < 1024:
                sub_in_channels = -1
                sub_out_channels = out_channels//2
            elif in_channels == 2048:
                sub_in_channels = 2048
                sub_out_channels = 64    
            elif out_channels >= 2048:
                sub_in_channels = 512
                sub_out_channels = out_channels//2
            else:
                sub_in_channels = 512
                sub_out_channels = -1
            self.conv = Conv2dSplitChannels(in_channels*batch_size if bs_as_group and batch_size != 1 else in_channels, 
                                              out_channels*batch_size if bs_as_group and batch_size != 1 else out_channels, 
                                              kernel_size, 
                                              sub_in_channels , 
                                              sub_out_channels, 
                                              stride, 
                                              padding, 
                                              dilation, 
                                              groups if not bs_as_group else batch_size, 
                                              bias=bias, 
                                              activation=act
                                              )
        else:
            self.conv = Conv2dSplit(in_channels*batch_size if bs_as_group and batch_size != 1 else in_channels,
                                    out_channels*batch_size if bs_as_group and batch_size != 1 else out_channels, 
                                    kernel_size, 
                                    stride, 
                                    padding,
                                    dilation,
                                    groups if not bs_as_group else batch_size,
                                    bias=bias,
                                    activation=act)
        self.conv.config.deallocate_activation = deallocate_input
        self.conv.config.shard_layout = shard_layout
        
    def process_state_dict_from_torch(self, state_dict: dict, prefix: str):
        if not self.preprocess_state_dict:
            return
        conv_name = 'conv'
        conv_weight = state_dict.pop(f'{prefix}weight')
        state_dict[f'{prefix}{conv_name}.weight'] = conv_weight
        if f'{prefix}bias' in state_dict:
            conv_bias = state_dict.pop(f'{prefix}bias')
            state_dict[f'{prefix}{conv_name}.bias'] = conv_bias
        input_shape = state_dict.pop(f'{prefix}input_shape')
        state_dict[f'{prefix}{conv_name}.input_shape'] = input_shape
        output_shape = state_dict.pop(f'{prefix}output_shape')
        state_dict[f'{prefix}{conv_name}.output_shape'] = output_shape
        ## 
        if self.bs_as_group and self.bs != 1:
            conv_weight = conv_weight.repeat(self.bs, 1, 1, 1)
            state_dict[f'{prefix}{conv_name}.weight'] = conv_weight
            if f'{prefix}{conv_name}.bias' in state_dict:
                conv_bias = conv_bias.repeat(self.bs)
                state_dict[f'{prefix}{conv_name}.bias'] = conv_bias
        
    def prehook_load_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.process_state_dict_from_torch(state_dict, prefix)
        super().prehook_load_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, inp, divisor=3, threshold=3*500*400, split_padding = False):
        if self.bs_as_group: 
            if inp.shape[-1] != self.bs * self.in_channels:
                B, H, W, C = inp.shape
                assert B*C == self.bs * self.in_channels
                inp = inp.permute(0, 3, 1, 2).reshape((1, B*C, H, W)).permute(0, 2, 3, 1)
            output = self.conv(inp, divisor=divisor, threshold=threshold, split_padding=split_padding)
            _, H, W, Co = output.shape
            output = output.permute(0, 3, 1, 2).reshape([B, Co//B, H, W])
            output = output.permute(0, 2, 3, 1)
            return output
        else:
            return self.conv(inp, divisor=divisor, threshold=threshold)
    
class ConvModule_(op.BaseModule):
    """A conv block that bundles conv/norm/activation layers.
    """
    
    _abbr_ = 'conv_block'
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 bn=False,
                 activation=None,
                 inplace=True,
                ):
        super(ConvModule_, self).__init__()

        # build convolution layer
        self.conv = op.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        
        # build activation layer
        if activation is not None:
            assert activation == 'relu', "Only ReLU is supported for now. Activation registered is not enable yet"
        self.activate = op.Functional(ttnn.relu) if activation == 'relu' else op.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x
    