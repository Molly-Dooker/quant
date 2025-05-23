from typing import List, Tuple, Set, Dict, Union
import numpy as np
import ttnn
import torch
from bos_metal import device_box

# TTNN types 
list_ttnn_tensor = List[ttnn.Tensor]
tuple_ttnn_tensor = Tuple[ttnn.Tensor]
set_ttnn_tensor = Set[ttnn.Tensor]
dict_ttnn_tensor = Dict[str, ttnn.Tensor]
ttnn_tensors_type = Union[
    ttnn.Tensor, 
    list_ttnn_tensor, 
    tuple_ttnn_tensor, 
    set_ttnn_tensor, 
    dict_ttnn_tensor
]

# Torch types
list_torch_tensor = List[torch.Tensor]
tuple_torch_tensor = Tuple[torch.Tensor]
set_torch_tensor = Set[torch.Tensor]
dict_torch_tensor = Dict[str, torch.Tensor]
torch_tensors_type = Union[
    torch.Tensor, 
    list_torch_tensor, 
    tuple_torch_tensor, 
    set_torch_tensor, 
    dict_torch_tensor
]

def is_numpy_array(a):
    return type(a).__module__ == np.__name__

def is_tensor(tensor):
    return isinstance(tensor, (ttnn.Tensor, torch.Tensor)) or is_numpy_array(tensor)    

def from_torch_one(tensor, device=None) -> ttnn.Tensor:
    device = device_box.resolve(device)
    assert is_tensor(tensor), f"to_torch_one: tensor must be ttnn.Tensor or torch.Tensor, got {type(tensor)}"
    if isinstance(tensor, ttnn.Tensor):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        to_dtype = ttnn.bfloat16
        # if int from torch, convert to bfloat16 in ttnn, because ttnn.uint* for positive ints
        return ttnn.from_torch(tensor, dtype=to_dtype, device=device)
    elif is_numpy_array(tensor):
        return from_torch_one(torch.from_numpy(tensor), device=device)

def from_torch_many(tensor, device=None) -> ttnn_tensors_type:
    device = device_box.resolve(device)
    if is_tensor(tensor):
        return from_torch_one(tensor, device=device)
    elif isinstance(tensor, (list, set, tuple)):
        return [from_torch_many(t, device=device) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: from_torch_many(t, device=device) for k, t in tensor.items()}
    else:
        # raise ValueError(f'Tensor type is not valid, got {type(tensor)}')
        return tensor

def to_torch_one(tensor) -> torch.Tensor:
    assert is_tensor(tensor), f"to_torch_one: tensor must be ttnn.Tensor or torch.Tensor, got {type(tensor)}"
    if isinstance(tensor, torch.Tensor):
        return tensor
    elif isinstance(tensor, ttnn.Tensor):
        to_dtype = torch.float32
        return ttnn.to_torch(tensor, dtype=to_dtype)
    else:
        return tensor
    
def to_torch_many(tensor) -> torch_tensors_type:
    if is_tensor(tensor):
        return to_torch_one(tensor)
    elif isinstance(tensor, (list, set, tuple)):
        return [to_torch_many(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: to_torch_many(t) for k, t in tensor.items()}
    else:
        # raise ValueError(f'Tensor type is not valid, got {type(tensor)}')
        return tensor
