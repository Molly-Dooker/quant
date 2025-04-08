import ipdb
import sys
import argparse
import math
import itertools
from transformers import DetrImageProcessor, DetrForObjectDetection
from loguru import logger
from transformers.utils.fx import symbolic_trace
import torch
from PIL import Image
import requests
import numpy as np
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from datasets import load_dataset
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics
from _util import class_names, keyword_to_itype, update_stats, get_preds, transform, custom_collate_fn, fold_frozen_bn_to_identity
from _quanto import _quantize, _Calibration, _requantize
from safetensors.torch import load_file, save_file
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
    quantize_activation
)
from optimum.quanto.nn import QConv2d, QLinear, QModuleMixin
import os
import json



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

def eval(model, device, dataloader, processor, prefix=''):
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    metrics = DetMetrics()
    metrics.names = class_names
    model.to(device)
    model.eval()  
    with torch.no_grad():
        total_batch = len(dataloader)
        for idx, batch in enumerate(tqdm(dataloader,desc='EVAL..')):
            logger.bind(file_only=True).info(f'eval... {idx+1}/{total_batch}')
            objects = objects = batch.pop('objects')
            inputs = {
                'pixel_values': batch.pop('pixel_values').to(device),
                'pixel_mask': batch.pop('pixel_mask').to(device)}
            outputs = model(**inputs)               
            origin_shape = torch.stack([torch.tensor(shape_) for shape_ in batch['origin_shape']])
            results = processor.post_process_object_detection(outputs, target_sizes=origin_shape, threshold=0.001)
            preds = get_preds(results)
            stats = update_stats(preds, objects, stats, device)
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
    stats.pop("target_img", None)
    if len(stats):
        metrics.process(**stats)
    result = metrics.results_dict
    mAP50   = result['metrics/mAP50(B)'].item()
    mAP5095 = result['metrics/mAP50-95(B)'].item()
    logger.info(f'{prefix} mAP50:{mAP50 : .3f}, mAP50-95 : {mAP5095:.3f}')


def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()  
    iter = math.ceil(num/dataloader.batch_size)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating...")):
            logger.bind(file_only=True).info(f'calibrate... {idx+1}/{iter}')
            inputs = {
                'pixel_values': batch.pop('pixel_values').to(device),
                'pixel_mask': batch.pop('pixel_mask').to(device)}
            _ = model(**inputs)               


def main(args):
    logger_enable(args.prefix)
    logger.info('start!')
    EVAL = args.eval
    STAT = args.stat
    if not EVAL:

        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        if args.size==-1:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm")
        else:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm", size={"height": args.size, "width": args.size})
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        model = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model)
        

        model2 = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model2)
        state_dict = load_file(f'{args.saveroot}/default_quant.safetensors')
        with open(f'{args.saveroot}/default_quant_map.json', 'r') as f:
            loaded_quantization_map = json.load(f)        
        _requantize(model2, state_dict, loaded_quantization_map)
        freeze(model2)

        RMSE = dict()
        data= dict()
        def prehook(module, inputs):
            activation = inputs[0].detach().cpu()
            data[module.name]={'input':activation}
        def hook(module, inputs, outputs):            
            data[module.name]['output']=outputs.detach().cpu()

        model.to('cuda:2')
        model.eval()  
        for name, m in model.named_modules():
            if isinstance(m,(nn.Conv2d, nn.Linear)):
                m._hooks=dict()
                m.name = name
                m._hooks['prehook'] = m.register_forward_pre_hook(prehook)
                m._hooks['hook']    = m.register_forward_hook(hook)
        model2.to('cuda:3')
        model2.eval()  
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='calibrate'):
                inputs = {
                    'pixel_values': batch.pop('pixel_values').to('cuda:2'),
                    'pixel_mask': batch.pop('pixel_mask').to('cuda:2')}
                _ = model(**inputs)    
                for name, m in model2.named_modules():
                    if isinstance(m,(QConv2d, QLinear)):                        
                        input = data[name]['input'].to('cuda:3')
                        output = m(input)
                        output_ = data[name]['output'].to('cuda:3')
                        rmse = torch.sqrt(torch.mean((output-output_)**2)).detach().cpu().item()
                        RMSE.setdefault(name, []).append(rmse)
                ipdb.set_trace()
        ipdb.set_trace()



        eval(model, args.device, dataloader, processor, 'quantized')

        #  디폴트 의 conv2d linear 에 hook 과 prehook 을 넣어서 각 input 과 output 을 잡아서 모델 이름에 대해 dict에 넣음
        # quant 모델을 순환하며 dict 의 keyname 과 같은 레이어에 input을 넣고 output을 가져와서 rms 함 


        # base model evaluation
        # eval(model,args.device,dataloader,processor,'default')
        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)
        exclude = []
        if args.exclude is not None:
            exclude = [ x for x in args.exclude.replace(' ','').split(',') ]    
        logger.info(f'exclude : {exclude}')        
        # exclude = ['class_labels_classifier',
        #            'bbox_predictor.layers.0',
        #            'bbox_predictor.layers.1',
        #            'bbox_predictor.layers.2',
        # ]
        _quantize(model, weights=weights, activations=activations, exclude=exclude) # custom quantize       
        if activations is not None:
            logger.info('Calibrate start...')
            # with Calibration(): 
            with _Calibration(): # custom Calibration
                calibrate(model, args.device, dataloader,10)
        logger.info('frozen model')    
        freeze(model)
        eval(model, args.device, dataloader, processor, 'quantized')
        os.makedirs(args.saveroot,exist_ok=True)
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        # qmap 저장하기
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'w') as f:
            json.dump(quantization_map(model), f)
        logger.info('end!')
    if EVAL:
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        if args.size==-1:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm")
        else:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm", size={"height": args.size, "width": args.size})
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

        model_reloaded = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model_reloaded)
        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'r') as f:
            loaded_quantization_map = json.load(f)
        requantize(model_reloaded, state_dict, loaded_quantization_map, args.device)
        freeze(model_reloaded)
        logger.info("Serialized quantized model")
        eval(model_reloaded, args.device, dataloader, args.size, 'reloaded')
        logger.info("end!")
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="detr")
    parser.add_argument("--prefix", type=str, default="test1")
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--dataset_name", type=str, default='rafaelpadilla/coco2017')
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/COCO')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=int, default=1, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--stat', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-stat', dest='stat', action='store_false', help='Disable stat mode')

    parser.add_argument("--size", type=int, default=800)
    parser.add_argument('--exclude', type=str, required=False, help='')
    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)

    
