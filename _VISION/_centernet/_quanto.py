
from typing import Any, Dict, List, Optional, Union
import torch
from torch.nn import functional as F, init
from optimum.quanto.nn import QModuleMixin, quantize_module
from optimum.quanto.tensor import Optimizer, qtype
from optimum.quanto.quantize import _quantize_submodule, set_module_by_name
from optimum.quanto.nn import QLinear, QConv2d
from optimum.quanto.tensor.activations import ActivationQBytesTensor, quantize_activation
from typing import Optional
from torch.nn.modules.module import  register_module_forward_hook, register_module_forward_pre_hook
from torch.overrides import TorchFunctionMode
from optimum.quanto import absmax_scale, QTensor
from optimum.quanto.calibrate import _updated_scale
import types
import re
import ipdb




def _quantize_input(module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
    input = input[0]
    if isinstance(input, ActivationQBytesTensor):
        if input.qtype != module.activation_qtype:
            raise ValueError(
                "Models with heterogeneous quantized activations are not supported:"
                f" expected {module.activation_qtype.name} input but got {input.qtype.name} instead."
            )
    else:
        input = quantize_activation(input, qtype=module.activation_qtype, scale=module.input_scale)
    return input


def _save_to_state_dict(self, destination, prefix, keep_vars):
    if self.weight_qtype is None or not self.frozen:
        # Save standard weight Tensor
        destination[prefix + "weight"] = (
            self.weight if (self.weight is None or keep_vars) else self.weight.detach()
        )
    else:
        # Save QTensor using dedicated method
        self.weight.save_to_state_dict(destination, prefix + "weight.", keep_vars)
    if self.bias is not None:
        destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
    destination[prefix + "input_scale"] = self.input_scale if keep_vars else self.input_scale.detach()
    # destination[prefix + "output_scale"] = self.output_scale if keep_vars else self.output_scale.detach()



def is_match(name, patterns):
    for pattern in patterns:
        if pattern.startswith('re:'):
            regex = pattern[3:]
            if re.match(regex, name): 
                return True
        else:
            if pattern==name:
                return True
    return False

def _quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude

    for name, m in model.named_modules():
        if include is not None:
            if is_match(name,include):
                _quantize_submodule(model, name, m, weights=weights, activations=activations, optimizer=optimizer)
                # convtransepose2d 케이스 추가
                if isinstance(m,torch.nn.ConvTranspose2d):
                    qmodule = QConvTranspose2d.from_module(m, weights=weights, activations=activations, optimizer=optimizer)            
                    set_module_by_name(model, name, qmodule)
                    qmodule.name = name
                    for name, param in m.named_parameters():
                        setattr(m, name, None)
                        del param
            continue

        if isinstance(m,torch.nn.LayerNorm): continue
        if is_match(name,exclude): continue
        _quantize_submodule(model, name, m, weights=weights, activations=activations, optimizer=optimizer)

        # convtransepose2d 케이스 추가
        if isinstance(m,torch.nn.ConvTranspose2d):
            qmodule = QConvTranspose2d.from_module(m, weights=weights, activations=activations, optimizer=optimizer)            
            set_module_by_name(model, name, qmodule)
            qmodule.name = name
            for name, param in m.named_parameters():
                setattr(m, name, None)
                del param

        

    # 전체적으로  output quantizer 제거
    # conv2d 는 input quantizer 붙임
    for name, m in model.named_modules():
        if not isinstance(m,QModuleMixin): continue
        m.disable_output_quantization()
        m._quantize_hooks.pop("output", None)
        del m._buffers["output_scale"]
        if isinstance(m, QConv2d): m._quantize_hooks["input"] = m.register_forward_pre_hook(_quantize_input)        
        m._save_to_state_dict = types.MethodType(_save_to_state_dict, m)

class _Calibration(TorchFunctionMode):
    """A custom torch dispatch mode to calibrate quantized modules.

    In order to improve the accuracy of the quantized activations, the input and output
    scales of each quantized module are evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.

    The dispatch mode also tracks the calls to each torch function down the model graph, and applies optional
    optimizations:
    - streamline: do not quantize activations that are immediately consumed by an incompatible function (like `add` or `silu`).

    Args:
        momentum (`float`): the momentum to use when updating scales.
        streamline (`bool`): if True, avoid quantizing activations when they are consumed by an incompatible function. Defaults to True.
        debug (`bool`): provide very verbose feedback on the console during calibration.
    """

    def __init__(self, *args, momentum: float = 0.9, streamline=True, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.streamline = streamline
        if streamline:
            self.modules_qactivations = {}
            self.streamline_hooks = {}
        self.debug = debug

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        qinput = QTensor in types
        output = func(*args, **kwargs)
        if self.streamline and qinput:
            for i, arg in enumerate(args):
                module = getattr(arg, "src_module", None)
                if module is not None:
                    if isinstance(output, ActivationQBytesTensor):
                        # Quantized activations are required for that module
                        self.modules_qactivations[module] = True
                    elif isinstance(output, torch.Tensor):
                        # Quantized activations are not required for that module unless another function requires them
                        qactivations_required = self.modules_qactivations.get(module, False)
                        self.modules_qactivations[module] = qactivations_required
        return output

    def __enter__(self):
        super().__enter__()
        self.pre_handle = register_module_forward_pre_hook(self.calibrate_input)
        # self.post_handle = register_module_forward_hook(self.calibrate_output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.pre_handle.remove()
        # self.post_handle.remove()
        if self.streamline:
            for handle in self.streamline_hooks.values():
                handle.remove()

    def calibrate_input(self, module: torch.nn.Module, input_, momentum: float = 0.9):
        """Calibrate a module input scale

        This is registered as a global hook that is called before any module forward pre hook.
        """
        if isinstance(module, QModuleMixin) and module.activation_qtype is not None:
            input = input_[0]
            if isinstance(input, ActivationQBytesTensor):
                # Just adopt the maximum scale of the input
                module.input_scale = torch.max(input._scale)
            else:
                # Evaluate the best scale
                input_scale = absmax_scale(input, module.activation_qtype)
                module.input_scale = _updated_scale(module.input_scale, input_scale, momentum)
            if self.streamline and module not in self.streamline_hooks:
                # Add a hook to tag the module outputs (after the module quantization hook in QModuleMixin)
                self.streamline_hooks[module] = module.register_forward_hook(self.tag_outputs)
            if len(input_)==1:
                return input
            else:
                return input_

    def calibrate_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """Calibrate a module output scale

        This is registered as a global hook that is called before any module forward hook.

        When the module is a QModuleMixin, its outputs are not quantized yet because they
        are only quantized in the QModuleMixin.quantize_output forward hook.
        """
        if isinstance(module, (QModuleMixin)) and module.activation_qtype is not None:
            # Evaluate the optimal scale per-tensor and update output scale
            output_scale = absmax_scale(output, module.activation_qtype, axis=None)
            module.output_scale = _updated_scale(module.output_scale, output_scale, self.momentum)
            return output
        else:
            if self.streamline:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin) and child.activation_qtype is not None:
                        qactivations_required = self.modules_qactivations.get(child, False)
                        if not qactivations_required:
                            # Disable output quantization for this child as its outputs are only consumed by incompatible functions.
                            child.disable_output_quantization()
            if self.debug:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin):
                        classname = child.__class__.__name__
                        trace = f"{name}({classname}) activations are"
                        if child.activation_qtype is None:
                            trace += " not quantized."
                        else:
                            trace += f" quantized to {child.activation_qtype} with scale {child.output_scale}."
                        print(trace)

    def tag_outputs(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """Mark outputs as generated by a module

        This is called as a module forward hook that is called after the QModuleMixin.quantize_output
        forward hook.

        This is useful in streamline mode to identify the module that generated a specific QTensor.
        """
        output.src_module = module


def _requantize(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    quantization_map: Dict[str, Dict[str, str]],
    device: torch.device = None,
):
    if device is None:
        device = next(model.parameters()).device
        if device.type == "meta":
            device = torch.device("cpu")

    # Quantize the model with parameters from the quantization map
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)

    # Move model parameters and buffers to CPU before materializing quantized weights
    for name, m in model.named_modules():

        def move_tensor(t, device):
            if t.device.type == "meta":
                return torch.empty_like(t, device=device)
            return t.to(device)

        for name, param in m.named_parameters(recurse=False):
            setattr(m, name, torch.nn.Parameter(move_tensor(param, "cpu")))
        for name, param in m.named_buffers(recurse=False):
            setattr(m, name, move_tensor(param, "cpu"))
    for name, m in model.named_modules():
        if not isinstance(m,QModuleMixin): continue
        m.disable_output_quantization()
        m._quantize_hooks.pop("output", None)
        del m._buffers["output_scale"]
        if isinstance(m, QConv2d): m._quantize_hooks["input"] = m.register_forward_pre_hook(_quantize_input)        
        m._save_to_state_dict = types.MethodType(_save_to_state_dict, m)

    # Move to target device
    model.to(device)
    # Load the quantized model weights
    model.load_state_dict(state_dict, strict=False)




class QConvTranspose2d(QModuleMixin, torch.nn.ConvTranspose2d):
    @classmethod
    def qcreate(
        cls,
        module,
        weights: qtype,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            dtype=module.weight.dtype,
            device=device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        return F.conv_transpose2d(
            input,
            self.qweight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )








