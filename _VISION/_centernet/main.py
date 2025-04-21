import ipdb
import sys
import argparse
import math
import itertools
import os
import json
import ipdb.stdout
from loguru import logger

import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from _util import fold_frozen_bn_to_identity, eval_transform, _collate_fn_eval, keyword_to_itype
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        gt_path = os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json')
        cocoGt = COCO(gt_path)        
        model.to(device)
        model.eval()       
        all_results = []
        all_targets = []
        with torch.no_grad():
            for src, targets, metas in tqdm(dataloader,desc='eval...'):
                src = src.to(device)
                output = model(src)
                hm  = output['hm'].sigmoid_()
                wh  = output['wh']
                reg = output['reg'] #if self.opt.reg_offset else None
                # if self.opt.flip_test:
                #     hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                #     wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                #     reg = reg[0:1] if reg is not None else None
                # forward_time = time.time()    
                # results = processor.post_process_object_detection(outputs, target_sizes=Size, threshold=0.001)
                results = processor(hm, wh, reg, metas)
                all_results.extend(results)
                all_targets.extend(targets)

        coco_results = []
        for idx, output in enumerate(all_results):
            if len(all_targets[idx]) > 0 and 'image_id' in all_targets[idx][0]:
                image_id = all_targets[idx][0]['image_id']
            else:
                continue

            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            boxes  = output['boxes'].detach().cpu().numpy()  # [x1, y1, x2, y2]
            for score, label, box in zip(scores, labels, boxes):
                x1, y1, x2, y2 = box
                bbox = [x1, y1, x2 - x1, y2 - y1]
                result = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(b) for b in bbox],
                    "score": float(score)
                }
                coco_results.append(result)

        cocoDt = cocoGt.loadRes(coco_results)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP5095 = cocoEval.stats[0].item(); mAP50 = cocoEval.stats[1].item(); 
        logger.info(f'{prefix} mAP50:{mAP50 : .3f}, mAP50-95 : {mAP5095:.3f}')


def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()  
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating...")):
            _ = model(batch[0].to(device))               


from _centernet import Config, CenterNet
from utils.functions import ctdet_decode
import cv2
import sys
from PIL import Image
import numpy as np

label2id = {
    'N/A': 83,
    'airplane': 5,
    'apple': 53,
    'backpack': 27,
    'banana': 52,
    'baseball bat': 39,
    'baseball glove': 40,
    'bear': 23,
    'bed': 65,
    'bench': 15,
    'bicycle': 2,
    'bird': 16,
    'boat': 9,
    'book': 84,
    'bottle': 44,
    'bowl': 51,
    'broccoli': 56,
    'bus': 6,
    'cake': 61,
    'car': 3,
    'carrot': 57,
    'cat': 17,
    'cell phone': 77,
    'chair': 62,
    'clock': 85,
    'couch': 63,
    'cow': 21,
    'cup': 47,
    'dining table': 67,
    'dog': 18,
    'donut': 60,
    'elephant': 22,
    'fire hydrant': 11,
    'fork': 48,
    'frisbee': 34,
    'giraffe': 25,
    'hair drier': 89,
    'handbag': 31,
    'horse': 19,
    'hot dog': 58,
    'keyboard': 76,
    'kite': 38,
    'knife': 49,
    'laptop': 73,
    'microwave': 78,
    'motorcycle': 4,
    'mouse': 74,
    'orange': 55,
    'oven': 79,
    'parking meter': 14,
    'person': 1,
    'pizza': 59,
    'potted plant': 64,
    'refrigerator': 82,
    'remote': 75,
    'sandwich': 54,
    'scissors': 87,
    'sheep': 20,
    'sink': 81,
    'skateboard': 41,
    'skis': 35,
    'snowboard': 36,
    'spoon': 50,
    'sports ball': 37,
    'stop sign': 13,
    'suitcase': 33,
    'surfboard': 42,
    'teddy bear': 88,
    'tennis racket': 43,
    'tie': 32,
    'toaster': 80,
    'toilet': 70,
    'toothbrush': 90,
    'traffic light': 10,
    'train': 7,
    'truck': 8,
    'tv': 72,
    'umbrella': 28,
    'vase': 86,
    'wine glass': 46,
    'zebra': 24
}


coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def _postprocessor(hm,wh,reg,metas, cat_spec_wh, K, scale, post0, post1, post2):
    dets = post0(hm, wh, reg=reg, cat_spec_wh=cat_spec_wh, K=K) #ctdet_decode
    results = []
    for det, meta in zip(dets, metas):
        # 배치당 
        scores = []; boxes =[]; labels=[];
        det = det.unsqueeze(0)
        det1 = post1(det, meta, scale) #Ctdet.post_process
        det2 = post2(det1)
        for cls in range(1,81): # coco 80 class 
            if det2[cls].shape[0]==0: continue
            box = det2[cls][:,:4] #xyxy 방식
            score = det2[cls][:,-1]
            label = label2id[coco_class_name[cls-1]]
            label = torch.tensor([label]*box.shape[0])
            scores.append(score)
            boxes.append(box)
            labels.append(label)
        boxes = torch.tensor(np.vstack(boxes))
        scores = torch.tensor(np.concat(scores))
        labels = torch.tensor(np.concat(labels))
        result = {'scores':scores, 'labels':labels, 'boxes':boxes}
        results.append(result)
    return results
    
        


            





    return dets_

def main(args):
    logger_enable(args.prefix)
    # logger.info('start!')
    EVAL = args.eval
    if not EVAL:

        torch.manual_seed(0)
        opt = Config(load_model='_model/ctdet_coco_dla_2x.pth', device=args.device)       
        Ctdet = CenterNet(opt)
        # img = cv2.imread("im1.jpg")
        # Ctdet.model.eval()
        # Ctdet.model.to(args.device)
        # Ctdet.run(img)     
        # return
        img_dir = os.path.join(args.coco_dir,'images', 'val2017')
        ann_file = os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json')
        preprocessor = lambda image : Ctdet.pre_process(image, 1.0 ,None)    
        dataset = CocoDetection(root=img_dir, annFile=ann_file, transforms=lambda img, target : eval_transform(img, target, preprocessor))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn_eval, num_workers=args.num_workers)

        model = Ctdet.model
        processor = lambda hm,wh,reg,metas : _postprocessor(hm, wh, reg ,metas, cat_spec_wh=False, K=100, scale=1.0, post0=ctdet_decode, post1= Ctdet.post_process, post2 = Ctdet.merge_outputs)
        eval(model, args.device, dataloader, processor, 'default')

                



        return


        model = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model)
        processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm", size={"height": args.size, "width": args.size})

        img_dir = os.path.join(args.coco_dir,'images', 'val2017')
        ann_file = os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json')
        dataset = CocoDetection(root=img_dir, annFile=ann_file, transforms=lambda img, target : eval_transform(img, target, processor))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn_eval, num_workers=args.num_workers)
        # base model evaluation
        if args.default: eval(model, args.device, dataloader, processor, 'default')

        weights = keyword_to_itype(args.weights)
        activations = keyword_to_itype(args.activations)
        exclude = ['class_labels_classifier', 're:^bbox_predictor.*', 're:^model.backbone.conv_encoder.model.encoder.stages.0.layers.0.*', 're:^model.encoder.layers.2.self_attn.*', 're:^model.encoder.layers.5.self_attn.*']
        if args.exclude is not None:
            exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
            if args.exclude=='': exclude = []
        logger.info(f'exclude : {exclude}')        
        _quantize(model, weights=weights, activations=activations, exclude=exclude) # custom quantize
        if activations is not None:
            with _Calibration(): # custom Calibration
                calibrate(model, args.device, dataloader)
        freeze(model)
        eval(model, args.device, dataloader, processor, 'quantized')
        os.makedirs(args.saveroot,exist_ok=True)
        save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
        # qmap 저장하기
        with open(f'{args.saveroot}/{args.prefix}.json', 'w') as f:
            json.dump(quantization_map(model), f)
        # logger.info('end!')
    if EVAL:
        if args.size==-1:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm")
        else:
            processor = DetrImageProcessor().from_pretrained(args.model_name, revision="no_timm", size={"height": args.size, "width": args.size})
        img_dir = os.path.join(args.coco_dir,'images', 'val2017')
        ann_file = os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json')
        dataset = CocoDetection(root=img_dir, annFile=ann_file, transforms=lambda img, target : eval_transform(img, target, processor))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn_eval, num_workers=args.num_workers)
        model = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model)

        state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
        with open(f'{args.saveroot}/{args.prefix}.json', 'r') as f:
            qmap = json.load(f)
        _requantize(model, state_dict, qmap, args.device)
        freeze(model)
        eval(model, args.device, dataloader, processor, 'reloaded')
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="centernet")
    parser.add_argument("--prefix", type=str, default="centernet1")
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--coco_dir", type=str, default='/Data/Dataset/coco')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=1, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--stat', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-stat', dest='stat', action='store_false', help='Disable stat mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    parser.add_argument("--size", type=int, default=800)
    parser.add_argument('--exclude', type=str)


    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)

    
