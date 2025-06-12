import torch
from torch.ao.quantization import (
    QConfigMapping,
    QConfig,
    default_per_channel_weight_observer,
    HistogramObserver
)
from ._qconfig_mapping import _get_default_qconfig_mapping


from typing import Any, Callable, Dict, List, Tuple, Union

qconfig = QConfig(
    activation=HistogramObserver.with_args(reduce_range=False,qscheme=torch.per_tensor_symmetric),
    weight=default_per_channel_weight_observer)


simple_qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear,qconfig).set_object_type(torch.nn.Conv2d,qconfig)

deault_qconfig_mapping = _get_default_qconfig_mapping(is_qat=False,backend='x86',version=0)



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