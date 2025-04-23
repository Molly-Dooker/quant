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
from _util import eval_transform, _collate_fn_eval, keyword_to_itype, _postprocessor, refacor_deformconv, _quantize_deformconv
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
    quantize_activation,    
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from _centernet import CenterNet, Config, DeformConv, DeformConv2
from utils.functions import ctdet_decode
from _centernet import deformconv2d
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

        result = {'mAP50:95':f'{cocoEval.stats[0].item():.5f}',
                  'mAP50':f'{cocoEval.stats[1].item():.5f}',
                  'mAP75':f'{cocoEval.stats[2].item():.5f}',
                  'mAP50:95-s':f'{cocoEval.stats[3].item():.5f}',
                  'mAP50:95-m':f'{cocoEval.stats[4].item():.5f}',
                  'mAP50:95-l':f'{cocoEval.stats[5].item():.5f}',
                  'mAR50:95-1':f'{cocoEval.stats[6].item():.5f}',
                  'mAR50:95-10':f'{cocoEval.stats[7].item():.5f}',
                  'mAR50:95-100':f'{cocoEval.stats[8].item():.5f}',
                  'mAR50:95-s':f'{cocoEval.stats[9].item():.5f}',
                  'mAR50:95-m':f'{cocoEval.stats[10].item():.5f}',
                  'mAR50:95-l':f'{cocoEval.stats[11].item():.5f}',                  
                  }
        for key, value in result.items():
            logger.info(f'{prefix} {key:12} : {value}')


def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()  
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating...")):
            _ = model(batch[0].to(device))               



def main(args):
    LAYERS = ['base.base_layer.0', 'base.level0.0', 'base.level1.0', 'base.level2.tree1.conv1', 'base.level2.tree1.conv2', 'base.level2.tree2.conv1', 'base.level2.tree2.conv2', 'base.level2.root.conv', 'base.level2.project.0', 'base.level3.tree1.tree1.conv1', 'base.level3.tree1.tree1.conv2', 'base.level3.tree1.tree2.conv1', 'base.level3.tree1.tree2.conv2', 'base.level3.tree1.root.conv', 'base.level3.tree1.project.0', 'base.level3.tree2.tree1.conv1', 'base.level3.tree2.tree1.conv2', 'base.level3.tree2.tree2.conv1', 'base.level3.tree2.tree2.conv2', 'base.level3.tree2.root.conv', 'base.level3.project.0', 'base.level4.tree1.tree1.conv1', 'base.level4.tree1.tree1.conv2', 'base.level4.tree1.tree2.conv1', 'base.level4.tree1.tree2.conv2', 'base.level4.tree1.root.conv', 'base.level4.tree1.project.0', 'base.level4.tree2.tree1.conv1', 'base.level4.tree2.tree1.conv2', 'base.level4.tree2.tree2.conv1', 'base.level4.tree2.tree2.conv2', 'base.level4.tree2.root.conv', 'base.level4.project.0', 'base.level5.tree1.conv1', 'base.level5.tree1.conv2', 'base.level5.tree2.conv1', 'base.level5.tree2.conv2', 'base.level5.root.conv', 'base.level5.project.0', 'base.fc', 'dla_up.ida_0.proj_1.conv_offset_mask', 'dla_up.ida_0.proj_1.deformconv2d', 'dla_up.ida_0.up_1', 'dla_up.ida_0.node_1.conv_offset_mask', 'dla_up.ida_0.node_1.deformconv2d', 'dla_up.ida_1.proj_1.conv_offset_mask', 'dla_up.ida_1.proj_1.deformconv2d', 'dla_up.ida_1.up_1', 'dla_up.ida_1.node_1.conv_offset_mask', 'dla_up.ida_1.node_1.deformconv2d', 'dla_up.ida_1.proj_2.conv_offset_mask', 'dla_up.ida_1.proj_2.deformconv2d', 'dla_up.ida_1.up_2', 'dla_up.ida_1.node_2.conv_offset_mask', 'dla_up.ida_1.node_2.deformconv2d', 'dla_up.ida_2.proj_1.conv_offset_mask', 'dla_up.ida_2.proj_1.deformconv2d', 'dla_up.ida_2.up_1', 'dla_up.ida_2.node_1.conv_offset_mask', 'dla_up.ida_2.node_1.deformconv2d', 'dla_up.ida_2.proj_2.conv_offset_mask', 'dla_up.ida_2.proj_2.deformconv2d', 'dla_up.ida_2.up_2', 'dla_up.ida_2.node_2.conv_offset_mask', 'dla_up.ida_2.node_2.deformconv2d', 'dla_up.ida_2.proj_3.conv_offset_mask', 'dla_up.ida_2.proj_3.deformconv2d', 'dla_up.ida_2.up_3', 'dla_up.ida_2.node_3.conv_offset_mask', 'dla_up.ida_2.node_3.deformconv2d', 'ida_up.proj_1.conv_offset_mask', 'ida_up.proj_1.deformconv2d', 'ida_up.up_1', 'ida_up.node_1.conv_offset_mask', 'ida_up.node_1.deformconv2d', 'ida_up.proj_2.conv_offset_mask', 'ida_up.proj_2.deformconv2d', 'ida_up.up_2', 'ida_up.node_2.conv_offset_mask', 'ida_up.node_2.deformconv2d', 'hm.0', 'hm.2', 'wh.0', 'wh.2', 'reg.0', 'reg.2']
    index = int(args.device[-1])
    start = index*11
    stop  = (index + 1) * 11
    LAYERS_=[m for m in LAYERS[start:stop]]
    
    EVAL = args.eval

    if not EVAL:
        
        for layer in tqdm(LAYERS_):
            logger_enable(layer)
            opt = Config(load_model='_model/ctdet_coco_dla_2x.pth', device=args.device)       
            Ctdet = CenterNet(opt)
            model = Ctdet.model 
            # modify deformconv
            refacor_deformconv(model) # fold batchnorm while refactoring deformconv (deformconv+bn case)
            dummy_input = torch.randn(1, 3, 512, 512)
            model.eval()
            fold_all_batch_norms(model.base, dummy_input.shape, dummy_input=dummy_input) # fold batchnorm for model.base (conv2d+bn case)


            img_dir = os.path.join(args.coco_dir,'images', 'val2017')
            ann_file = os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json')
            preprocessor = lambda image : Ctdet.pre_process(image, 1.0 ,None)    
            dataset = CocoDetection(root=img_dir, annFile=ann_file, transforms=lambda img, target : eval_transform(img, target, preprocessor))
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn_eval, num_workers=args.num_workers)
            processor = lambda hm,wh,reg,metas : _postprocessor(hm, wh, reg ,metas, cat_spec_wh=False, K=100, scale=1.0, post0=ctdet_decode, post1= Ctdet.post_process, post2 = Ctdet.merge_outputs)
            if args.vanilla:  eval(model, args.device, dataloader, processor, 'default')
            weights = keyword_to_itype(args.weights)
            activations = keyword_to_itype(args.activations)
            # exclude = ['re:^hm.*', 're:^wh.*', 're:^reg.*']
            # if args.exclude is not None:
            #     exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
            #     if args.exclude=='': exclude = []
            include = layer
            logger.info(f'include : {include}')          
            _quantize(model, weights=weights, activations=activations, include= include) # nn.Linear, nn.Conv2d + nn.ConvTranspose2d
            _quantize_deformconv(model, weights=weights, activations=activations, include=include) # deformconv

            # 해당 모델에 대해 deformconv 까지 
            if activations is not None:
                with _Calibration(): # custom Calibration
                    calibrate(model, args.device, dataloader,1000)
            freeze(model)
            eval(model, args.device, dataloader, processor, 'quantized')
            # os.makedirs(args.saveroot,exist_ok=True)
            # save_file(model.state_dict(), f'{args.saveroot}/{args.prefix}.safetensors')
            # # qmap 저장하기
            # with open(f'{args.saveroot}/{args.prefix}.json', 'w') as f:
            #     json.dump(quantization_map(model), f)
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

    parser.add_argument('--vanilla', action='store_true', help='')
    parser.add_argument('--no-vanilla', dest='default', action='store_false')

    parser.add_argument("--size", type=int, default=800)
    parser.add_argument('--exclude', type=str)


    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)

    
