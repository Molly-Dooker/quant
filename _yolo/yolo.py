import sys
import ipdb
import math
import itertools
from loguru import logger
from tqdm import tqdm

import torch
import numpy as np;
from datasets import load_dataset
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from torch.fx import symbolic_trace

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

from util import class_names, keyword_to_itype, update_stats
from _yolov8s import Yolov8s
from _dataloader import Processor, transform, custom_collate_fn



        

def eval(model, device, dataloader, prefix=''):    
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
            stats = update_stats(preds, objects, origin_shapes, stats, device)
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

if __name__ == '__main__':
    logger_enable('yolo8s')
    device = 'cuda:3'     
    yolo =  YOLO("yolov8s.pt")
    yolo.fuse()
    yolo.eval()

    ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO',split='val')
    processor = Processor()
    prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
    model = Yolov8s(yolo.model.model)
    eval(model,device,dataloader,'default')



    weights = keyword_to_itype('int8')
    activations = keyword_to_itype('int8')
    exclude = []
    quantize(model, weights=weights, activations=activations, exclude=exclude)

    if activations is not None:
        print('Calibrate start...')
        with Calibration():
            calibrate(model, device, dataloader)
    print("frozen model")
    freeze(model)

    eval(model,device,dataloader,'quantized')