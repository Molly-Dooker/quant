from ultralytics import YOLO
from torch.fx import symbolic_trace
import torch
import ipdb;
from PIL import Image
from yolo import Yolov8s
from datasets import load_dataset
from _dataloader import Processor, transform, custom_collate_fn
from _nms import _get_nms 
import numpy as np;
import cv2
from tqdm import tqdm
from loguru import logger
import sys
import evaluate
import json
from torchmetrics.detection import MeanAveragePrecision
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

def eval(model, device, dataloader,get_nms, prefix=''):
    model.to(device)
    model.eval()
    # ipdb.set_trace()
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='EVAL..'):
            img = batch['image'].to(device)
            objects = batch['objects']
            id  = batch['image_id']            
            origin_shapes = batch['origin_shape']
            preds = y8(img)
            nms = get_nms(preds,origin_shapes)
            ipdb.set_trace()

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def get_coco_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    logger_enable('yolo8s')
    device = 'cuda:2'  
    
    coco_gt = get_coco_gt('_coco_gt/coco_gt.json')

    yolo =  YOLO("yolov8s.pt")
    yolo.fuse()
    yolo.eval()

    ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO',split='val')   

    processor = Processor()
    prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    y8 = Yolov8s(yolo.model.model)

    # get_nms = lambda preds,origin_shapes: _get_nms(preds,origin_shapes,conf_thres=0.05,iou_thres=0.45,scaled_shape=(640,640))
    get_nms = lambda preds,origin_shapes: _get_nms(preds,origin_shapes,conf_thres=0.25,iou_thres=0.7,scaled_shape=(640,640))
    eval(y8,device,dataloader,get_nms)

    # yolo.to(device)
    # ipdb.set_trace()
    # result_origin = yolo.predict(source=ds[:32]['image'], save=True)    
    # ipdb.set_trace()
    




