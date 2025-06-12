import torch
from torch.nn import ReLU
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d
from typing import Any, Callable, Dict, Union
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization import (
    QConfigAny,
    QConfigMapping,
    QConfig,
    default_per_channel_weight_observer,
    HistogramObserver,
    get_default_qat_qconfig,
    default_placeholder_observer,
    default_reuse_input_qconfig,
    default_quint8_weight_qconfig,
    FixedQParamsFakeQuantize,
    default_weight_fake_quant,
    default_weight_observer,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer        
)

import contextlib


_FIXED_QPARAMS_OP_TO_OBSERVER: Dict[Union[Callable, str], _PartialWrapper] = {
    torch.nn.Hardsigmoid: default_fixed_qparams_range_0to1_observer,
    torch.nn.functional.hardsigmoid: default_fixed_qparams_range_0to1_observer,
    "hardsigmoid": default_fixed_qparams_range_0to1_observer,
    "hardsigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Sigmoid: default_fixed_qparams_range_0to1_observer,
    torch.sigmoid: default_fixed_qparams_range_0to1_observer,
    "sigmoid": default_fixed_qparams_range_0to1_observer,
    "sigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Softmax: default_fixed_qparams_range_0to1_observer,
    torch.nn.Tanh: default_fixed_qparams_range_neg1to1_observer,
    torch.tanh: default_fixed_qparams_range_neg1to1_observer,
    "tanh": default_fixed_qparams_range_neg1to1_observer,
    "tanh_": default_fixed_qparams_range_neg1to1_observer,
}



def _get_default_qconfig_mapping(
    is_qat: bool, backend: str, version: int
) -> QConfigMapping:
    """
    Return the default QConfigMapping for the given quantization type and backend.
    """
    if is_qat:
        qconfig = get_default_qat_qconfig(backend, version)
    else:
        # this is difference
        qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=False,qscheme=torch.per_tensor_symmetric),
            weight=default_per_channel_weight_observer
            )
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer

    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer.
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend in ("fbgemm", "x86"):
        qconfig_transpose = QConfig(
            activation=qconfig.activation, weight=default_weight
        )
    else:
        qconfig_transpose = qconfig

    # currently layernorm only supports float weights
    # we have to add this because otherwise there will be a extra quantize-dequantize pair
    qconfig_layernorm = QConfig(
        activation=qconfig.activation, weight=default_placeholder_observer
    )

    qconfig_mapping = (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type("reshape", default_reuse_input_qconfig)
        .set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose)
        .set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose)
        .set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose)
        .set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm)
        .set_object_type(torch.nn.LayerNorm, qconfig_layernorm)
        .set_object_type(torch.nn.PReLU, default_quint8_weight_qconfig)
    )
    # Use special observers for ops with fixed qparams
    fixed_qparams_observer_to_qconfig: Dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(
                activation=activation, weight=default_weight
            )
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)

    # TODO Currently it's required that separate ops in a fused op/module have the same qconfig.
    #      Need to be able to support fusion of ops with different qconfigs

    return qconfig_mapping

qconfig = QConfig(
    activation=HistogramObserver.with_args(reduce_range=False,qscheme=torch.per_tensor_symmetric),
    weight=default_per_channel_weight_observer)


simple_qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear,qconfig).set_object_type(torch.nn.Conv2d,qconfig).set_object_type(ConvReLU2d,qconfig).set_object_type(ReLU,qconfig)

default_qconfig_mapping = _get_default_qconfig_mapping(is_qat=False,backend='x86',version=0).set_object_type(torch.cat,None)





def save_graph(model, file_path = "model_graph"):

    # 표준 출력을 파일로 리디렉션합니다.
    with open(file_path, 'w') as f:
        with contextlib.redirect_stdout(f):
            # 이 블록 안에서의 print() 또는 stdout 출력은 모두 파일 'f'에 기록됩니다.
            model.graph.print_tabular()


# for node in model_.graph.nodes:
#     if not node.op == 'call_module': continue
#     if not node.target.startswith('activation_post'): continue
#     node_before = node.args[0]
#     node_after = node.users.keys()
#     obs = getattr(model_,node.target)
#     print(f'{node.target:20} min:{obs.min_val:2.4f} max:{obs.max_val:2.4f}')
#     print(f' - before : {node_before}')
#     print(f' - atfer  : {list(node_after)}')
#     print('-----------------------------------------')      


# with open("code_jit.py", "w") as f:
#     f.write(jit_model.code)