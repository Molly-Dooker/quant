import numpy as np
from typing import List, Tuple, Set, Dict, Union
from bos_metal import dirs, torch, helpers, op
import ttnn
from bos_metal import device_box

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

tensor_store_dir = dirs.models_gendir.make("bevformer/tensors")
tensor_ext = ".pt"

def load_tensor(name) -> torch.Tensor:
    print(f"Load tensor name {name}")
    name_ = name + tensor_ext
    return torch.load(tensor_store_dir.get_path(name_))

def save_tensor(tensor, name):
    print(f"Save tensor name {name}")
    name_ = name + tensor_ext
    print("Save tensor at", tensor_store_dir.get_path(name_))
    torch.save(tensor, tensor_store_dir.get_path(name_))

do_assert = True
raise_error = False
def assert_tensor(ttnn_tensor, torch_name, module):
    if not do_assert:
        return True, ""
    # assert not helpers.anyinf(ttnn_tensor)
    # print(f"Assert tensor name {torch_name}")
    torch_name = f"{torch_name}.pt"
    torch_path = tensor_store_dir.get_path(torch_name)
    if not torch_path.exists():
        msg = f"Mising torch tensor: {torch_name}"
        print(msg)
        return False, msg
    if module is not None and not isinstance(ttnn_tensor, torch.Tensor):
        to_torch_func = module.to_torch if isinstance(module, op.BaseModule) else module
        # print("Assert tensor, convert ttnn to torch tensor")
        ttnn_tensor: torch.Tensor = to_torch_func(ttnn_tensor)
    while ttnn_tensor.ndim < 4:
        ttnn_tensor = ttnn_tensor.unsqueeze(0)
    # print("Assert tensor, load and compare tensors")
    torch_tensor: torch.Tensor = torch.load(torch_path)
    while torch_tensor.ndim < 4:
        ttnn_tensor = ttnn_tensor.squeeze(0)
    passed, msg = helpers.compare_tensors(torch_tensor, ttnn_tensor, custom_msg=torch_name)

    if raise_error and not passed:
        raise ValueError(msg)
    return passed, msg


def get_list_shape(tensor):
    return [i for i in tensor.shape]

def add_dim(tensor, dim):
    tensor_shape = get_list_shape(tensor)
    for i in range(len(dim)):
        if dim[i]==0:
            dim[i] = tensor_shape.pop(0)
    return ttnn.reshape(tensor, dim)

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