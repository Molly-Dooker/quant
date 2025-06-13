import argparse
import sys
import itertools
import math
import ipdb
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, transform
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.quantization as quant
from torch.quantization import (
    FakeQuantize,
    FakeQuantizeBase,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    HistogramObserver,
    prepare_qat,
    convert,
)
import numpy as np
import evaluate
from accelerate import init_empty_weights
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    AutoImageProcessor,
)
from optimum.quanto import (
    QTensor,
    freeze,
    qfloat8,
    qint4,
    qint8,
    quantization_map
)
import os

from torch.fx import symbolic_trace, GraphModule
from torch_config.myconfig import simple_qconfig_mapping, default_qconfig_mapping
# from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import (
    prepare_fx, 
    convert_fx,
)
from _quanto import _quantize, _requantize, _Calibration
def logger_enable(prefix=''):
    def console_filter(record):
        # extra에 file_only가 True인 경우 콘솔 출력 제외
        return not record["extra"].get("file_only", False)
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add(sys.stdout, level="INFO", format=LOG_FORMAT, filter=console_filter)
    logger.add("_logs/log", rotation="500 MB", level="INFO", format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)




def eval(model, test_loader, prefix=''):

    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]

    metric = evaluate.load("accuracy")

    for batch in tqdm(test_loader,desc='eval...'):
        data, target = batch["pixel_values"].numpy(), batch["labels"].numpy()
        output = model.run(output_names,{input_name: data})[0]
        output = output.argmax(-1)
        metric.add_batch(predictions=output,references=target)

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, dataloader, samples=1000):
    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]
    iter =  min(math.ceil(samples/dataloader.batch_size), dataloader.__len__())
    for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
        data = batch["pixel_values"].numpy()
        output = model.run(output_names,{input_name: data})[0]

def calibrate_wrapper(model, samples, dataloader):
    calibrate(model,dataloader,samples)

    
    
    
    


import shutil
import onnxruntime as ort
from onnxsim import simplify
import onnx
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
def main(args):
    logger_enable(args.prefix)
    EVAL = args.eval
    if not EVAL : 
        
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dummy_input = torch.randn(1, 3, 224, 224)
        

      
        

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        filename = 'resnet18.onnx'
        model = onnx.load_model(filename)
        
        try:
            model, _ = simplify(model)
        except:
            print('ONNX Simplifier failed. Proceeding with unsimplified model')
        # ipdb.set_trace()
        # session = ort.InferenceSession(model.SerializeToString(),providers=providers)



        # eval(session,dataloader,args.prefix)
        # _ = fold_all_batch_norms_to_weight(model)


        sim = QuantizationSimModel(model=model,
                                quant_scheme=QuantScheme.post_training_tf,
                                default_activation_bw=8,
                                default_param_bw=8,
                                providers=providers,
                                config_file='_custom_config.json')


        root = f'output/{args.prefix}/'
        
        os.makedirs(root,exist_ok=True)
        shutil.copyfile('_custom_config.json',root+'congfig.json')

        with open(root+'graph.graph', "w") as f:
            f.write(str(sim.model.model.graph))
            
        sim.compute_encodings(forward_pass_callback=lambda session,samples : calibrate_wrapper(session,samples,dataloader),
                            forward_pass_callback_args=500)
        ipdb.set_trace()
        sim.export(path=root, filename_prefix='qq')



        
        
        # onnx.save(model,'resnet18_simple.onnx')
                    



        # model_ = prepare_fx(model,default_qconfig_mapping , dummy_input)
        # calibrate(model_,args.device,dataloader,500)                  
        # model_.to('cpu')
        # q_model = convert_fx(model_)  
        # ipdb.set_trace()

        # jit_model = torch.jit.trace(q_model, dummy_input) 

        
        
        # # weight 
        # model.fc.weight().int_repr()
        # model.fc.weight().dequantize()
        # model.fc.weight().q_per_channel_scales()
        # model.fc.weight().q_per_channel_zero_points()
        
        # from torch.ao.nn.quantized import Linear as qlinear

        






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="resnet")
    parser.add_argument("--prefix", type=str, default="resnet1")
    # parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=6, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    parser.add_argument('--exclude', type=str)

    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   

    main(args)