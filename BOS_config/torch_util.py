import contextlib
import torch
from torch.nn import ReLU
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d, ConvBnReLU2d
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
    default_fixed_qparams_range_neg1to1_observer,
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    get_default_qat_qconfig_mapping
)

qconfig = get_default_qat_qconfig()
simple_qconfig_mapping  = QConfigMapping().set_object_type(torch.nn.Linear,qconfig).set_object_type(torch.nn.Conv2d,qconfig).set_object_type(ReLU,qconfig).set_object_type(torch.nn.BatchNorm2d,qconfig)
default_qconfig_mapping = get_default_qat_qconfig_mapping().set_object_type(torch.cat,None)



def save_graph(model, file_path = "model_graph"):

    # 표준 출력을 파일로 리디렉션합니다.
    with open(file_path, 'w') as f:
        with contextlib.redirect_stdout(f):
            # 이 블록 안에서의 print() 또는 stdout 출력은 모두 파일 'f'에 기록됩니다.
            model.graph.print_tabular()