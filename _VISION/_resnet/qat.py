import argparse
import sys
import json
import itertools
import math
import ipdb
import os
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
import torch.quantization as tq
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)



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
from _quanto import _quantize, _requantize, _Calibration
from aimet_torch.batch_norm_fold import fold_all_batch_norms
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




def eval(model, device, test_loader, prefix=''):
    model.to(device)
    model.eval()
    metric = evaluate.load("accuracy")
    with torch.no_grad():
        for batch in tqdm(test_loader,desc='eval...'):
            data, target = batch["pixel_values"], batch["labels"]
            data= data.to(device)
            output = model(data)
            if isinstance(output, QTensor):
                output = output.dequantize()
            output = output.argmax(-1).cpu()
            metric.add_batch(predictions=output,references=target)

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            # ipdb.set_trace()
            data = batch["pixel_values"].to(device)
            _ = model(data)




act_observer = MovingAverageMinMaxObserver.with_args(
    quant_min=-128, quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric
)
wt_observer = MovingAveragePerChannelMinMaxObserver.with_args(
    quant_min=-128, quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric
)

learnable_act = LearnableFakeQuantize.with_args(
    ovserver = act_observer,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    use_grad_scaling=True,
)


learnable_wt = LearnableFakeQuantize.with_args(
    ovserver = wt_observer,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    use_grad_scaling=True,
)


# learnable_weights = lambda channels: LearnableFakeQuantize.with_args(

#     quant_min=-128,
#     quant_max=127,
#     dtype=torch.qint8,
#     qscheme=torch.per_channel_symmetric,
#     use_grad_scaling=True,
#     channel_len=channels,
# )


def main(args):
    logger_enable(args.prefix)
    EVAL = args.eval
    if not EVAL : 
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        fold_all_batch_norms(model, dummy_input.shape, dummy_input=dummy_input)
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        model.qconfig = None
        qconfig = tq.QConfig(
            activation=learnable_act,
            weight = learnable_wt
        )
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.qconfig = qconfig
        model.train()     
        ipdb.set_trace()   
        prepare_qat(model, inplace=True)
        ipdb.set_trace() 
        # activation 에 대한 fakequantizer 를 prehook으로 옮김.
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fq = m.activation_post_process
                m._forward_hooks.clear()  # 출력 후 fake-quant 제거
                m.register_forward_pre_hook(lambda mod, inp, fq=fq: (fq(inp[0]),))
        ipdb.set_trace()
        # model.to(args.device)
        # input = torch.randn(16,3,256,256,requires_grad=True,device=args.device)        
        # logit = model(input)



        if args.default: eval(model, args.device, dataloader, 'default')

        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)
        # make exclude list
        exclude = ['fc']
        if args.exclude is not None:
            exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
            if args.exclude=='': exclude = []
        logger.info(f'exclude : {exclude}')   
        # prepare model to quantize
        _quantize(model, weights=weights, activations=activations, exclude=exclude)
        if activations is not None:
            with _Calibration():
                calibrate(model, args.device, dataloader, 1000)
        print("frozen model")
        freeze(model)
        eval(model, args.device, dataloader,'quantized')        
        os.makedirs(args.saveroot,exist_ok=True)
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'w') as f:
            json.dump(quantization_map(model), f)

    if EVAL:
        processor = ViTImageProcessor.from_pretrained(args.model_name)
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'r') as f:
            qmap = json.load(f)
        config = ViTConfig.from_pretrained(args.model_name)
        with init_empty_weights():
            model = ViTForImageClassification.from_pretrained(args.model_name, config=config)
        _requantize(model, state_dict, qmap, args.device)
        freeze(model)
        eval(model, args.device, dataloader,'reloaded')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="resnet")
    parser.add_argument("--prefix", type=str, default="resnet1")
    # parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=1, help="The device to use for evaluation.")
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