
import onnx
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig, QuantizationConfig
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForImageClassification
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)
from optimum.onnxruntime.utils import evaluation_loop
from datasets import load_dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

import  torch
import numpy as np
from ultralytics.data.augment import LetterBox
from _dataloader import Processor, custom_collate_fn, _transform
import ipdb


quantizer = ORTQuantizer.from_pretrained('.')


qconfig = QuantizationConfig(
is_static             = True,
format                = QuantFormat.QDQ,
mode                  = QuantizationMode.QLinearOps,
activations_dtype     = QuantType.QInt8,
weights_dtype         = QuantType.QInt8,
per_channel           = True,
reduce_range          = False,
operators_to_quantize = ["MatMul", "Conv", "Gemm"],
)

dataset = load_dataset(path='rafaelpadilla/coco2017', cache_dir='/Data/Dataset/COCO', split='val')

calibration_ds = dataset.select_columns('image').with_transform(lambda batch: _transform(batch, Processor())).shuffle(seed=42).select(range(200))



calibration_config =  AutoCalibrationConfig.percentiles(calibration_ds)



ranges = quantizer.fit(
        dataset = calibration_ds,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
        use_gpu=True
)


ipdb.set_trace()