import argparse
import sys
import json
import itertools
import math
from loguru import logger
from tqdm import tqdm
from util import ipdb_sys_excepthook
# ipdb_sys_excepthook()
import torch
import torch.nn.functional as F
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
    Calibration,
    QTensor,
    freeze,
    qfloat8,
    qint4,
    qint8,
    quantization_map,
    quantize,
    requantize,
)
import ipdb
# f
# from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
from optimum.quanto.nn.qlinear import QLinear
# optimum.quanto.nn.qlinear.QLinear


def logger_enable(prefix=''):
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add("_logs/log",
            rotation="500 MB",
            level="INFO",
            format=LOG_FORMAT)

    # 콘솔 로거
    logger.add(sys.stdout,
            level="INFO",
            format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)




def test(model, device, test_loader, prefix=''):
    model.to(device)
    model.eval()
    metric = evaluate.load("accuracy")
    with torch.no_grad():
        for batch in tqdm(test_loader,desc='TEST..'):
            data, target = batch["pixel_values"], batch["labels"]
            data= data.to(device)
            output = model(data).logits
            if isinstance(output, QTensor):
                output = output.dequantize()
            output = output.argmax(-1).cpu()
            metric.add_batch(predictions=output,references=target)

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, device, test_loader, num=10000):
    model.to(device)
    model.eval()
    iter = math.ceil(num/test_loader.batch_size)
    with torch.no_grad():
        for batch in tqdm(itertools.islice(test_loader, iter), total=iter, desc="calibrating..."):
            # ipdb.set_trace()
            data = batch["pixel_values"].to(device)
            _ = model(data)


def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def transform(data_batch, processor):
    IMAGE = data_batch["image"]
    IMAGE = [image.convert('RGB') for image in IMAGE]
    inputs = processor(IMAGE, return_tensors="pt")
    inputs["labels"] = data_batch["label"]
    return inputs

def main(args):
    logger_enable(args.prefix)
    logger.info(f'START!')
    EVAL = args.eval

    if not EVAL : 
        processor = ViTImageProcessor.from_pretrained(args.model_name)
        model = ViTForImageClassification.from_pretrained(args.model_name)

        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True)

        # print("Model before quantization...")
        # test(model, args.device, test_loader, 'base')

        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)

        # make exclude list
        exclude = ['vit.encoder.layer.5.output.dense',
                'vit.encoder.layer.9.attention.attention.query',
                'vit.layernorm']
        for i in range(12):
            exclude.append(f'vit.encoder.layer.{i}.layernorm_after')
            exclude.append(f'vit.encoder.layer.{i}.layernorm_before')
        # prepare model to quantize
        quantize(model, weights=weights, activations=activations, exclude=exclude)
        # print(model)  
        if activations is not None:
            print('Calibrate start...')
            with Calibration():
                calibrate(model, args.device, dataloader)

        print("frozen model")
        freeze(model)
        test(model, args.device, dataloader)

        # Serialize model to a state_dict, save it to disk and reload it
        # weight 저장하기
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        # qmap 저장하기
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'w') as f:
            json.dump(quantization_map(model), f)


    if EVAL:
        processor = ViTImageProcessor.from_pretrained(args.model_name)
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True)

        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'r') as f:
            quantization_map = json.load(f)
        config = ViTConfig.from_pretrained(args.model_name)
        with init_empty_weights():
            model_reloaded = ViTForImageClassification.from_pretrained(args.model_name, config=config)
        requantize(model_reloaded, state_dict, quantization_map, args.device)
        freeze(model_reloaded)
        print("Serialized quantized model")
        # ipdb.set_trace()
        test(model_reloaded, args.device, dataloader)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ViT")
    parser.add_argument("--prefix", type=str, default="ViT2")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=int, default=3, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')
    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   

    main(args)