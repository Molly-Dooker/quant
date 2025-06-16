import ipdb
import sys
import math
import itertools
from loguru import logger
from tqdm import tqdm
import argparse
import json
import os

import torch
from datasets import load_dataset
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.ops import non_max_suppression

from safetensors.torch import load_file, save_file
from optimum.quanto import (
    freeze,
    quantization_map,
)
from _quanto import _quantize, _requantize, _Calibration
from _util import class_names, keyword_to_itype, update_stats
from _yolov8s import Yolov8s
from _dataloader import Processor, transform, custom_collate_fn


from torch_config.myconfig import simple_qconfig_mapping, default_qconfig_mapping, save_graph
from torch.ao.quantization.quantize_fx import (
    prepare_qat_fx,
    convert_fx,
    
    )
from torch.ao.quantization import prepare, convert


        

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
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            img = batch['image'].to(device)
            _   = model(img)

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

def main(args):
    logger_enable(args.prefix)
    logger.info('start!')
    EVAL = args.eval

    if not EVAL:
        yolo =  YOLO(args.model_name)
        yolo.fuse()
        yolo.eval()
        model = Yolov8s(yolo.model.model, args.size)
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        processor = Processor(new_shape=(args.size, args.size))
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers)
        # base model evaluation
        # eval(model, args.device, dataloader, args.size, 'quantized')
        dummy_input = torch.randn(1, 3, args.size, args.size)  
        model.eval()     

        
        
        qconfig_mapping = simple_qconfig_mapping.set_module_name('m22.dfl.conv',None)
        model_ = prepare_qat_fx(model, qconfig_mapping, dummy_input)
        calibrate(model_, args.device, dataloader)     
        model_.to('cpu')
        q_model = convert_fx(model_)
        save_graph(q_model)
        jit_model = torch.jit.trace(q_model, dummy_input) 
        ipdb.set_trace()
        eval(jit_model,'cpu', dataloader, args.size, 'quantized') 

        
        


    if EVAL:
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        processor = Processor(new_shape=(args.size, args.size))
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers)        
        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'r') as f:
            qmap = json.load(f)
        yolo =  YOLO(args.model_name)
        yolo.fuse()
        yolo.eval()
        model = Yolov8s(yolo.model.model, args.size)
        _requantize(model, state_dict, qmap, args.device)
        freeze(model)
        eval(model, args.device, dataloader, args.size, 'reloaded')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="yolo")
    parser.add_argument("--prefix", type=str, default="YOLO_tt")
    parser.add_argument("--model_name", type=str, default="yolov8s.pt")
    parser.add_argument("--dataset_name", type=str, default='rafaelpadilla/coco2017')
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/COCO')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=5, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])

    parser.add_argument("--size", type=int, default=416)
    parser.add_argument('--exclude', type=str)

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)