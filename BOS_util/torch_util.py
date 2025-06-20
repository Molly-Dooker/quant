from torch.ao.quantization import QConfig, HistogramObserver, default_per_channel_weight_observer, QConfigMapping
import torch


sym_qconfig = QConfig(
        activation=HistogramObserver.with_args(dtype= torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False),
        weight=default_per_channel_weight_observer,
)


asym_qconfig = QConfig(
        activation=HistogramObserver.with_args(dtype= torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False),
        weight=default_per_channel_weight_observer,
)


simple_sym_qconfig_mapping = QConfigMapping()\
        .set_object_type(torch.nn.Linear,sym_qconfig)\
        .set_object_type(torch.nn.Conv2d,sym_qconfig)\
        .set_object_type(torch.nn.ReLU,sym_qconfig)\
        .set_object_type(torch.nn.BatchNorm2d,sym_qconfig)\
        .set_object_type(torch.nn.LayerNorm,None)
        
        
simple_asym_qconfig_mapping = QConfigMapping()\
        .set_object_type(torch.nn.Linear,asym_qconfig)\
        .set_object_type(torch.nn.Conv2d,asym_qconfig)\
        .set_object_type(torch.nn.ReLU,asym_qconfig)\
        .set_object_type(torch.nn.BatchNorm2d,asym_qconfig)\
        .set_object_type(torch.nn.LayerNorm,None)