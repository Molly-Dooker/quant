import ipdb
import sys
import argparse
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

def logger_enable(prefix=''):
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add("_logs/log",
            rotation="500 MB",
            level="INFO",
            format=LOG_FORMAT)
    logger.add(sys.stdout,
            level="INFO",
            format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)

def eval(model, device, dataloader, processor, prefix=''):
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    metrics = DetMetrics()
    metrics.names = class_names
    model.to(device)
    model.eval()  
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='EVAL..'):
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

def main(args):
    logger_enable(args.prefix)
    logger.info('start!')
    EVAL = args.eval

    if not EVAL:
        ds = load_dataset(path='rafaelpadilla/coco2017', cache_dir='/Data/Dataset/COCO', split='val')
        processor = DetrImageProcessor().from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        ipdb.set_trace()
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        fold_frozen_bn_to_identity(model)

        # base model evaluation
        eval(model,args.device,dataloader,processor,'default')



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="detr")
    parser.add_argument("--prefix", type=str, default="detr")
    # parser.add_argument("--model_name", type=str, default="yolov8s.pt")
    # parser.add_argument("--dataset_name", type=str, default='rafaelpadilla/coco2017')
    # parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/COCO')
    parser.add_argument("--saveroot", type=str, default='./_model')
    # parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=int, default=2, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')
    parser.add_argument("--size", type=int, default=800, choices=[800])
    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)

    
