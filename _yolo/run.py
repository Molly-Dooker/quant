import ipdb.stdout
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
from torch import BoolTensor, IntTensor, Tensor
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

def get_preds(nms):
    preds=[]
    for nms_ in nms:
        pred = dict()        
        nms_ = nms_.cpu()
        pred['boxes']=nms_[:,:4]
        pred['scores']=nms_[:,4]
        pred['labels']=nms_[:,-1].to(torch.int32)
        preds.append(pred)
    return preds

def get_targets(objects):
    targets=[]
    for object in objects:
        target = dict()
        bboxes = object['bbox']
        boxes = []
        for bbox in bboxes:
            xyxy = torch.tensor([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            boxes.append(xyxy)
        if len(boxes)==0: 
            boxes = torch.tensor([])
            labels= torch.tensor([],dtype=torch.int32)
        else:
            boxes = torch.stack(boxes)
            labels = torch.tensor(object['label'],dtype=torch.int32)-1
        target['boxes']=boxes;
        target['labels']=labels;
        targets.append(target)
    return targets;

    

def eval(model, device, dataloader,get_nms, prefix=''):
    model.to(device)
    model.eval()
    # ipdb.set_trace()

    with torch.no_grad():
        for batch in tqdm(dataloader,desc='EVAL..'):
            img = batch['image'].to(device)
            objects = batch['objects']
            # id  = batch['image_id']            
            origin_shapes = batch['origin_shape']
            output  = model(img)
            nms     = get_nms(output,origin_shapes)
            preds   = get_preds(nms)            
            targets = get_targets(objects)           




def get_coco_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    logger_enable('yolo8s')
    device = 'cuda:2'  
    
    yolo =  YOLO("yolov8s.pt")
    yolo.fuse()
    yolo.eval()

    yolo.to(device)
    results = yolo.val(data="_coco.yaml", imgsz=640, plots=False)
    # result_origin = yolo.predict(source=ds[:32]['image'], save=True)    
    ipdb.set_trace()
    




