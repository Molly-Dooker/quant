from typing import Dict, Union, Tuple
import ttnn
from bos_metal import op
from tt.modules.ops import run, ConvModule
from .res_layer import ResLayer, BasicBlock, Bottleneck
from bevformer.utils import assert_many, load_many



blocks_type = Union[BasicBlock, Bottleneck]
arch_settings_type = Dict[int, Tuple[blocks_type, Tuple[int, int, int, int]]]

class ResNet(op.BaseModule):
    """ResNet backbone.
    """
    arch_settings: arch_settings_type = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 avg_down=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                ):
        super(ResNet, self).__init__()
        # Assertions
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        stem_channels = stem_channels or base_channels

        # Set attributes
        self.depth = depth
        self.stem_channels = stem_channels 
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        assert num_stages >= 1 and num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages
        self.style = style
        self.avg_down = avg_down
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        # Make layers
        self.conv1 = ConvModule(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True,
            act="relu",
            preprocess_state_dict=True,
            deallocate_input=True,
        )
        self.maxpool = op.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Res layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2**i
            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                dcn=dcn,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion *\
            base_channels *\
            2**(len(self.stage_blocks) - 1)

    def split_batch(self, x, op, divisor=6, threshold=3*400*500):
        B, C, H, W = x.shape
        assert B % divisor == 0, f"Batch size must be divisible by {divisor}"

        n_batch = B // divisor
        for i in range(divisor):
            # assume x is in shape of (6, 3, 900, 1600)
            x1 = x[i*n_batch:(i+1)*n_batch, :, :, :]
            x1 = run(x1, op, threshold=threshold)

            if i==0:
                ret = x1
            else:
                ret = ttnn.concat([ret, x1], 0)
                # ret = ttnn.reallocate(ret)
                # ttnn.deallocate(x1)
    
        # ttnn.deallocate(x)
        return ret
    
    def forward(self, x):
        """Forward function."""

        # Conv1
        print("Resnet conv1")
        x = self.conv1(x, threshold=3*550*500, divisor = 1)
        # assert_many(ttnn.to_torch(ttnn.permute(x, [0,3,1,2])), "resnet.conv1")
        
        

        print("Resnet maxpool")
        x = self.split_batch(x, self.maxpool, threshold=3*1100*1300, divisor=6)    # 6, 225, 400, 64
        # assert_many(x.permute(0,3,1,2), "resnet.maxpool")
        outs = []
        
        # Layers
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            print(f"Running res_layer {i}")
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        Bottleneck.count = 0
        return tuple(outs)
