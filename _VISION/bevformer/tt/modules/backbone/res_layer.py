from typing import Union
import torch
import torch.nn as nn
import ttnn
from bos_metal import op

from tt.modules.ops import ConvModule, ModulatedDeformConv2dPack
from bevformer.utils import save_tensor, assert_many, load_many, save_many


class BasicBlock(op.BaseModule):
    expansion = 1
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 **kwargs,
                ):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.dilation = dilation
        
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=True)
        self.conv2 = ConvModule(
            planes, 
            planes, 
            3, 
            padding=1, 
            bias=True, 
            activation='relu',
            preprocess_state_dict=True,    
        )
        self.downsample = downsample
        self.relu = ttnn.relu
        self.add = op.Add(deallocate_input=True)
    
    #TODO: config for Conv2dSplit
    def forward(self, x, devisor=3, threshold=3*450*400):
        """Forward function."""
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        self.add(out, identity)
        out = self.relu(out)

        return out

class Bottleneck(op.BaseModule):
    expansion = 4
    count = 0
    init = 0
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 *,
                 style='pytorch',
                 dcn=None,
                 deallocate_input=True,
                 **kwargs,
                ):
        super(Bottleneck, self).__init__()
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.deallocate_input = deallocate_input

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        # Conv1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=True,
            act="relu",
            deallocate_input=False,  # do not deallocate because it will discard identity
            split_channels=True if inplanes >= 2048 else False,  #in case in channels too large, split by in_channels    
            preprocess_state_dict=True,
            # bs_as_group=True if Bottleneck.init == 0 else False,
            # batch_size=6 if Bottleneck.init == 0 else 1
        )
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        
        # Conv2
        self.fallback_with_dcn = False
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = ConvModule(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=True,
                act="relu",
                deallocate_input=True,
                split_channels=True if planes >= 2048 else False,
                preprocess_state_dict=True,
                # bs_as_group=True if Bottleneck.init > 0 else False,
                # batch_size=6 if Bottleneck.init > 0 else 1
            )
        else:
            self.fallback_with_dcn = True
            self.conv2 = ModulatedDeformConv2dPack(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)

        # Conv3
        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=True,
            act="",
            deallocate_input=True,
            split_channels=True if planes * self.expansion >= 2048 else False,
            preprocess_state_dict=True,
            # bs_as_group=True if Bottleneck.init == 0 else False,
            # batch_size=6 if Bottleneck.init == 0 else 1
        )
        self.downsample = downsample
        self.relu = ttnn.relu
        self.add = op.Add(deallocate_input=True) # , activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]

        # Config for split conv
        self.flag = False
        if planes * self.expansion >= 2048:
            self.flag = True
            
        Bottleneck.init += 1
            
    def process_state_dict_from_torch(self, state_dict: dict, prefix: str):
        if self.fallback_with_dcn:
            del state_dict[f'{prefix}conv2.conv_offset.input_shape']
            del state_dict[f'{prefix}conv2.conv_offset.output_shape']
        
    def prehook_load_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.process_state_dict_from_torch(state_dict, prefix)
        super().prehook_load_state_dict(state_dict, prefix, *args, **kwargs)
        
    def forward(self, x):
        """Forward function."""
        print("Bottleneck running")

        def _inner_forward(x):
            identity = x
            print(f"Bottleneck {Bottleneck.count}.conv1")

            out = self.conv1(x,
                            divisor=1, 
                            threshold=3*1300*1100) # if not self.flag else 1459200) # 3*500*500
            # if Bottleneck.count in [15]:
                # assert_many(ttnn.to_torch(out.permute(0,3,1,2)), f"bottleneck_{Bottleneck.count}.conv1")
            if self.fallback_with_dcn:
                print("Start pytorch")
                out = ttnn.to_torch(out, dtype = torch.float)
                out = out.permute(0, 3, 1, 2)
                with torch.no_grad():
                    out = self.conv2(out)
                    out = self.bn2(out)
                print("End pytorch")
                out = ttnn.from_torch(
                    out.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
                )
                out = self.relu(out)
                out = ttnn.untilize(out)
            else:
                print(f"Bottleneck {Bottleneck.count}.conv2")
                # out = self.conv2(out, divisor=1, threshold=3*400*500 if Bottleneck.count > 13 else 3*1300*1100 if Bottleneck.count == 15 else 3*1000*1000 if Bottleneck.count== 7 else 3*400*500)
                out = self.conv2(out, divisor=1, threshold=3*500*500 if Bottleneck.count == 0  \
                                 else 3*700*700 if Bottleneck.count in range(1, 3) \
                                 else 3*1000*1000 if Bottleneck.count in range(3, 8) \
                                 else 3*700*1000 if Bottleneck.count in range(8, 14) \
                                 else 3*400*500
                                 )
                # if Bottleneck.count in [15]:
                    # assert_many(ttnn.to_torch(out.permute(0,3,1,2)), f"bottleneck_{Bottleneck.count}.conv2")
            print(f"Bottleneck {Bottleneck.count}.conv3")
            # out = self.conv3(out,
            #                 divisor=1,
            #                 threshold=3*500*500 if Bottleneck.count < 13 else 3*1000*1000 if Bottleneck.count== 7 else 3*400*500)
            out = self.conv3(out,
                            divisor=1,
                            threshold=3*500*500 if Bottleneck.count == 0 \
                                else 3*700*500 if Bottleneck.count in range(1, 13) \
                                else 3*700*1000
                                )
            # if Bottleneck.count in [15]:
                # assert_many(ttnn.to_torch(out.permute(0,3,1,2)), f"bottleneck_{Bottleneck.count}.conv3")
            if self.downsample is not None:
                B, H, W, C = x.shape
                divisor=2
                if C <= 512:
                    threshold = 3*700*500
                else:
                    threshold = 3*1300*1100
                print(f"Bottleneck {Bottleneck.count}.down")
                identity = self.downsample[0](identity, threshold=threshold, divisor=divisor)

            out = self.add(out, identity)
            return out

        out = _inner_forward(x)
        # if self.deallocate_input:
        #     ttnn.deallocate(x)
        
        out = self.relu(out)
        # assert_many(ttnn.to_torch(out).permute(0,3,1,2), f"bottleneck_{Bottleneck.count}.end")
        Bottleneck.count += 1

        return out

class ResLayer(op.Sequential):
    count = 0
    """ResLayer to build ResNet style backbone.

    Args:
        block (op.BaseModule): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 *,
                 style='pytorch',
                 downsample_first=True,
                 dcn=None,
                 **kwargs):
        self.block = block
        self.dcn = dcn
        self.downsample_first = downsample_first
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend([
                ConvModule(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=True,
                    act="",
                    deallocate_input=False,
                    split_channels=False if ResLayer.count < 3 else True,
                    preprocess_state_dict=True,
                    # bs_as_group=True if Bottleneck.init == 0 else False,
                    # batch_size=6 if Bottleneck.init == 0 else 1
                ),
            ])
            downsample = op.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    dilation=dilation,
                    style=style,
                    dcn=dcn,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dcn=dcn,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        dcn=dcn,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    dilation=dilation,
                    style=style,
                    dcn=dcn,
                    **kwargs
                )
            )
        super(ResLayer, self).__init__(*layers)

        ResLayer.count += 1
