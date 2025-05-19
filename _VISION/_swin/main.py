import argparse
import sys
import json
import itertools
import math
import ipdb
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, transform, update_statedict
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import evaluate
from accelerate import init_empty_weights
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
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
from _model import SwinTransformerV2, WindowAttention
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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

def main(args):
    logger_enable(args.prefix)
    EVAL = args.eval
    if not EVAL :
        model = SwinTransformerV2(img_size=256, window_size=8)
        # download from MS official github page
        # wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth
        state_dict = torch.load('_model/swinv2_tiny_patch4_window8_256.pth')['model']
        # qkv(Linear) in WindowAttention module will have its own bias
        state_dict = update_statedict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        total = 0
        for name, m in tqdm(model.named_modules()):
            if isinstance(m,(torch.nn.Linear,torch.nn.Conv2d)):
                param = m.weight.numel()
                total+=param


        processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        if args.default: eval(model, args.device, dataloader, 'default')
        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)
        # make exclude list
        exclude = ['head']
        if args.exclude is not None:
            exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
            if args.exclude=='': exclude = []
        logger.info(f'exclude : {exclude}')   
        # prepare model to quantize
        _quantize(model, weights=weights, activations=activations, exclude=exclude)
        if activations is not None:
            with _Calibration():
                calibrate(model, args.device, dataloader)
        freeze(model)
        eval(model, args.device, dataloader,'quantized')
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'w') as f:
            json.dump(quantization_map(model), f)

    if EVAL:
        processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)    
        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'r') as f:
            qmap = json.load(f)
        model = SwinTransformerV2(img_size=256, window_size=8)
        _requantize(model, state_dict, qmap, args.device)
        freeze(model)
        eval(model, args.device, dataloader,'reloaded')
        ipdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="swin")
    parser.add_argument("--prefix", type=str, default="swin")
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