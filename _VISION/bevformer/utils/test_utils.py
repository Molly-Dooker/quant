import warnings
from pathlib import Path
import numpy as np
import torch
import ttnn

def comp_pcc(golden, calculated, pcc=0.99):
    # golden = torch.Tensor(golden)
    # calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        print("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        print("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        print("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


def check_with_pcc_without_tensor_printout(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message

# ANSI escape codes for colors
class ANSI_COLORS:
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    YELLOW = '\033[93m'

class Validator:
    suffix = '.pth'
    
    def __init__(self, save_dir: str, pcc=0.98):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pcc_thred = pcc
        
    def _get_filename(self, filename):
        return filename if filename.endswith(self.suffix) else filename + self.suffix
    
    def _get_path(self, filename) -> Path:
        return self.save_dir/self._get_filename(filename)
    
    def save_tensor(self, tensor, filename):
        save_path = self._get_path(filename)
        torch.save(tensor, save_path)
        print(f"Tensor {filename} saved.")

    def load_tensor(self, filename):
        load_path = self._get_path(filename)
        if not load_path.exists():
            print(f"Tensor {filename} not found.")
            return None
        print(f"Loading tensor from {load_path}")
        return torch.load(load_path, map_location='cpu')
    
    def compare_tensors(self, tensor1, tensor2, name=None):
        if tensor1 is None and tensor2 is None:
            passing, pcc_msg = True, "Both tensors are None"
        if tensor1 is None or tensor2 is None:
            passing, pcc_msg = False, "One tensor is None"
        else:
            passing, pcc_msg = check_with_pcc_without_tensor_printout(tensor1.cpu(), tensor2.cpu(), pcc=self.pcc_thred)
        test_msg =  f"{ANSI_COLORS.GREEN}Test passed{ANSI_COLORS.RESET}" if passing else f"{ANSI_COLORS.RED}Test failed{ANSI_COLORS.RESET}"
        print_mgs = f"{name} | " if name else ""
        print_mgs += f"{pcc_msg} - {test_msg}" 
        print(print_mgs)
        return passing, pcc_msg

    def assert_tensor(self, tensor, filename, raise_error=False):
        # Load referenced tensor
        ref_tensor = self.load_tensor(filename)
        error_msg = ()

        # Assert values with PCC
        passing, pcc_msg = self.compare_tensors(ref_tensor, tensor, filename)
        if not passing:
            error_msg += (pcc_msg,)

        # Raise error if any
        if error_msg:
            if raise_error:
                raise AssertionError(error_msg)
            else:
                # print(error_msg) 
                pass    
            
validator = Validator('data/xtuan/test_outputs', 0.98)

load_tensor = validator.load_tensor 
assert_tensor = validator.assert_tensor
save_tensor = validator.save_tensor

def assert_many(tensors, name):
    # # Visualizer testing
    return
    if tensors is None:
        warnings.warn(f"Tensor for assertion is None for {name}")
        return
    if isinstance(tensors, ttnn.Tensor):
        validator.assert_tensor(ttnn.to_torch(tensors), name)
    elif isinstance(tensors, torch.Tensor):
        validator.assert_tensor(tensors, name)
    elif isinstance(tensors, (list, tuple, set)):
        for i, t in enumerate(tensors):
            assert_many(t, f'{name}.{i}')
    elif isinstance(tensors, dict):
        for key, t in tensors.items():
            assert_many(t, f'{name}.{key}')
    else:
        warnings.warn(f"Tensor type is not valid for assertion for {name}, got {type(tensors)}")

def save_many(tensors, name):
    # return
    if tensors is None:
        return
    if isinstance(tensors, (ttnn.Tensor, torch.Tensor)):
        validator.save_tensor(tensors, name)
    elif isinstance(tensors, (list, tuple, set)):
        for i, t in enumerate(tensors):
            save_many(t, f'{name}.{i}')
    elif isinstance(tensors, dict):
        for key, t in tensors.items():
            save_many(t, f'{name}.{key}')
    else:
        warnings.warn(f"Tensor type is not valid for saving for {name}, got {type(tensors)}")

def load_many(name):
    return load_tensor(name)