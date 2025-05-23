import bos_metal as bm
from bos_metal.models.backbones import BottleNeck
from bos_metal import helpers
import torch
import ttnn

def conv_2p(x1, x2, conv_obj):
        padding = conv_obj.padding
        conv_obj.padding = (0, 0)
        input_shape = torch.tensor(conv_obj.input_shape)
        orig_input_shape = input_shape.clone()
        output_shape = torch.tensor(conv_obj.output_shape)
        input_shape[0] = 1 
        input_shape[-1] = input_shape[-1]/2 + 2*padding[0]
        input_shape[-2] = input_shape[-2] + 2*padding[0]
        conv_obj.set_shapes(input_shape.tolist())
        #
        p1l = conv_obj(x1) # CONV
        x1.deallocate()
        output_shape[0] = 1
        output_shape = output_shape[[0,2,3,1]]
        #
        p1l = ttnn.reshape(p1l, output_shape.tolist())
        p1r = conv_obj(x2) # CONV
        x2.deallocate()
        p1r = ttnn.reshape(p1r, output_shape.tolist())
        p1 = ttnn.concat([p1l, p1r], 2)
        # 
        conv_obj.padding = padding
        conv_obj.set_shapes(orig_input_shape.tolist())
        return p1

def conv2d_split(x, conv_obj):
    for i in range(x.shape[0]):
        x_i = helpers.split_tensor(x[i:i+1, :,:,:], 2, conv_obj.padding)
        # print(f"Processing {i}:", x_i[0])
        x_i = conv_2p(x_i[0], x_i[1], conv_obj)
        if i == 0:
            out = x_i
        else:
            out = ttnn.concat([out, x_i], 0)
    
    return out




