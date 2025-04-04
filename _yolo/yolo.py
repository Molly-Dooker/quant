import sys
import ipdb
import math
import itertools
from loguru import logger
from tqdm import tqdm
import argparse
import json
import os

import torch
import numpy as np;
from datasets import load_dataset
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from torch.fx import symbolic_trace

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
)

from _util import class_names, keyword_to_itype, update_stats
from _yolov8s import Yolov8s
from _dataloader import Processor, transform, custom_collate_fn



        

def eval(model, device, dataloader, size=640, prefix=''):    
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    nms = lambda predictions: non_max_suppression(prediction=predictions, conf_thres=0.001, iou_thres=0.7, labels=[], nc=80, multi_label=True, agnostic=False, max_det=300, end2end=False, rotated=False)
    metrics = DetMetrics()
    metrics.names = class_names
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader,desc='EVAL..'):
            img = batch['image'].to(device)
            objects = batch['objects']
            # id  = batch['image_id']            
            origin_shapes = batch['origin_shape']
            output  = model(img)
            preds = nms(output)
            stats = update_stats(preds, objects, origin_shapes, stats, device, size)
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
        for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            img = batch['image'].to(device)
            _   = model(img)

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

def main(args):
    logger_enable(args.prefix)
    logger.info('start!')
    EVAL = args.eval

    if not EVAL:
        yolo =  YOLO(args.model_name)
        yolo.fuse()
        yolo.eval()

        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        processor = Processor(new_shape=(args.size, args.size))
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        model = Yolov8s(yolo.model.model, args.size)

        # check default performance
        eval(model, args.device, dataloader, args.size, 'default')
        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)
        exclude = []
        quantize(model, weights=weights, activations=activations, exclude=exclude)
        if activations is not None:
            print('Calibrate start...')
            with Calibration():
                calibrate(model, args.device, dataloader)
        print("frozen model")
        freeze(model)

        eval(model, args.device, dataloader, args.size, 'quantized')

        # Serialize model to a state_dict, save it to disk and reload it
        # weight 저장하기        
        os.makedirs(args.saveroot,exist_ok=True)
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        # qmap 저장하기
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'w') as f:
            json.dump(quantization_map(model), f)

    if EVAL:
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        processor = Processor(new_shape=(args.size, args.size))
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        
        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}_map.json', 'r') as f:
            loaded_quantization_map = json.load(f)

        yolo =  YOLO(args.model_name)
        yolo.fuse()
        yolo.eval()
        model_reloaded = Yolov8s(yolo.model.model, args.size)
        requantize(model_reloaded, state_dict, loaded_quantization_map, args.device)
        freeze(model_reloaded)
        print("Serialized quantized model")
        eval(model, args.device, dataloader, args.size, 'reloaded')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="yolo")
    parser.add_argument("--prefix", type=str, default="YOLOv8s")
    parser.add_argument("--model_name", type=str, default="yolov8s.pt")
    parser.add_argument("--dataset_name", type=str, default='rafaelpadilla/coco2017')
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/COCO')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=int, default=2, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')
    parser.add_argument("--size", type=int, default=640, choices=[640, 416])
    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)