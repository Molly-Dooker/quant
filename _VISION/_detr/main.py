import ipdb
import sys
import argparse
import math
import itertools
import os
import json
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
            for src, targets, Size in tqdm(dataloader,desc='eval...'):
                src = src.to(device)
                outputs = model(src)
                results = processor.post_process_object_detection(outputs, target_sizes=Size, threshold=0.001)
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


def main(args):
    logger_enable(args.prefix)
    # logger.info('start!')
    EVAL = args.eval
    if not EVAL:
        model = DetrForObjectDetection.from_pretrained(args.model_name, revision="no_timm")
        fold_frozen_bn_to_identity(model)
        ipdb.set_trace()
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
        
    parser = argparse.ArgumentParser(description="detr")
    parser.add_argument("--prefix", type=str, default="detr1")
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

    
