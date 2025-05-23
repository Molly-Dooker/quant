import pytest
import os
from pathlib import Path
import torch
import ttnn
import bos_metal as bm

from bos_metal import device_box, profiler, op, dirs, builder, helpers
from torchvision.models import resnet50
from resnet_config import resnet50_config_v55_4x5
from bos_metal.operations.conv import op_split, Conv2dSplit, run
from bos_metal.core import printv
from bos_models.bevformer.ttnn.utils import assert_tensor
# bm.RuntimeOptions.enable_verbose()
# bm.RuntimeOptions.enable_golden_test()
# bm.RuntimeOptions.enable_log_bhrc()



class ResnetModel(bm.models.Resnet):
    num_classes = 1000
    
    def __init__(self, **kwargs):
        super(ResnetModel, self).__init__(**kwargs)
        self.avgpool = op.AdaptiveAvgPool2d((1, 1))
        self.fc = op.Linear(512 * self.block_cls.expansion, self.num_classes)
        self.avgpool = torch.nn.Identity()
        self.fc = torch.nn.Identity()
        
        self.conv1 = Conv2dSplit(3, 
                                self.base_channels, 
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=True,  # fuse with batchnorm
                                activation='relu',
                                device=self.device)
        
    def split_batch(self, x, op, divisor=6, threshold=3*400*500):
        B, C, H, W = x.shape
        assert B % divisor == 0, f"Batch size must be divisible by {divisor}"

        n_batch = B // divisor
        for i in range(divisor):
            # assume x is in shape of (6, 3, 900, 1600)
            x1 = x[i*n_batch:(i+1)*n_batch, :, :, :]
            x1 = run(x1, op, threshold=threshold)
            op.output_shape = (6, C, H//2, W//2)
            op.input_shape = (6, C, H, W)          

            if i==0:
                ret = x1
            else:
                ret = ttnn.concat([ret, x1], 0)
                ttnn.deallocate(x1)
    
        ttnn.deallocate(x)
        return ret
        
    
    def forward(self, x):
        # Conv1
        x = self.conv1(x, threshold = 3*300*400)
        # assert_tensor(x, "resnet.conv1", self.conv1)  
        
        
        x = self.split_batch(x, self.maxpool, threshold=3*1200*1200)     
        # assert_tensor(x, "resnet.maxpool", self.maxpool)

        outputs = []
        channel_split = [144, 144, 144, 144]
        for id in range(1, self.num_stages + 1):
            res_layer = getattr(self, self._layer_name(id))
            for i, block in enumerate(res_layer):
                printv(f"Processing Backbone at {self._layer_name(id)}.{i}")
                x = block.forward_split(x, channel_split[id-1])
                # assert_tensor(x, "resnet.res_layer" + "_" + str(id - 1) + "." + str(i), block)
            assert_tensor(x, "resnet.res_layer" + "_" + str(id - 1), block)
            outputs = self._append_output(x, outputs, id-1)
            
        return [outputs]


@pytest.mark.resnet50_bos_a0_loop_b3_256x256
def resnet50_loop_256x256():
    # bm.RuntimeOptions.enable_runtime_analysis()
    device_box.open()

    # Load the model
    input_shape = (1, 3, 256, 256)
    input_tensor = torch.randn(input_shape)

    # Create the model
    torch_resnet50 = resnet50(weights=True)
    ttnn_resnet50 = ResnetModel()
    ttnn_resnet50.save_arch(dirs.this("resnet50_256x256.txt"))

    # Preprocess state dict
    processor = bm.ModelProcessor(torch_resnet50, capture_output=True)
    state_dict = processor.process_state_dict(input_tensor, save_to=dirs.this("resnet50_256x256"))
    ttnn_resnet50.load_state_dict(state_dict)
    ttnn_resnet50.load_config_dict(resnet50_config_v55_4x5)
    # ttnn_resnet50.save_config_dict(dirs.this("resnet50_256x256_config.json"))

    # Warm up:
    torch_output = processor.get_last_output_tensors()
    _ = ttnn_resnet50(input_tensor)

    # Loop
    # breakpoint()
    # try:
    #     for i in range(20):
    #         profiler.start("resnet50")
    #         _ = ttnn_resnet50(input_tensor)
    #         profiler.end("resnet50")
    # except:
    #     print(helpers.refine_tt_fatal())

    # Last output to torch 
    ttnn_out = ttnn_resnet50(input_tensor, to_torch=True)

    # Compare
    while torch_output.dim() < 4:
        torch_output = torch_output.unsqueeze(0)
    passed, msg = bm.compare_tensors(ttnn_out, torch_output)
    assert passed, msg

    profiler.print()
    device_box.close()
    
if __name__=="__main__":
    resnet50_loop_256x256()
