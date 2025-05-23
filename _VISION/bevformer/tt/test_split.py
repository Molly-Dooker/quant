import torch
import ttnn
from bos_metal import helpers, device_box

def split_tensor(x, num_splits, padding):
    #
    x_i = helpers.split_tensor(x, num_splits, padding)
    
    return x_i


tens = torch.randn(1, 256, 29, 50)
tens = ttnn.from_torch(torch.permute(tens, (0, 2, 3, 1)), device=device_box.get())

print(split_tensor(tens, 2, (1,1))[0].shape, split_tensor(tens, 2, (1,1))[1].shape)