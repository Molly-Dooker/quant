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

class TempLoggerPatch:
    """
    with 문 내에서 다른 모듈의 로거를 임시로 교체하는 컨텍스트 매니저.
    """
    def __init__(self, target_module, new_logger):
        self.target_module = target_module
        self.new_logger = new_logger
        self.original_logger = None

    def __enter__(self):
        """with 블록 시작 시 호출: 로거를 교체합니다."""
        # 원래 로거를 백업합니다.
        self.original_logger = getattr(self.target_module, 'logger', None)
        # 목표 모듈의 로거를 새로 설정한 로거로 교체합니다.
        self.target_module.logger = self.new_logger
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 블록 종료 시 호출: 로거를 원상 복구합니다."""
        # 백업해 둔 원래 로거로 복원합니다.
        self.target_module.logger = self.original_logger


        

def eval(model, dataloader, size=640, prefix=''):    
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    nms = lambda predictions: non_max_suppression(prediction=predictions, conf_thres=0.001, iou_thres=0.7, labels=[], nc=80, multi_label=True, agnostic=False, max_det=300, end2end=False, rotated=False)
    metrics = DetMetrics()
    metrics.names = class_names
    input_name = model.get_inputs()[0].name

    for batch in tqdm(dataloader,desc='EVAL..'):
        img = batch['image'].numpy()
        objects = batch['objects']
        # id  = batch['image_id']            
        origin_shapes = batch['origin_shape']
        output = model.run(None,{input_name:img})[0]
        preds = nms(torch.tensor(output))
        stats = update_stats(preds, objects, origin_shapes, stats, 'cpu', size)
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
    stats.pop("target_img", None)
    if len(stats):
        metrics.process(**stats)
    result = metrics.results_dict
    mAP50   = result['metrics/mAP50(B)'].item()
    mAP5095 = result['metrics/mAP50-95(B)'].item()
    logger.info(f'{prefix} mAP50:{mAP50 : .3f}, mAP50-95 : {mAP5095:.3f}')


def calibrate(model, dataloader, num=10000):
    # model.to(device)
    # model.eval()
    input_name = model.get_inputs()[0].name
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    # with torch.no_grad():
    for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
        img = batch['image'].numpy()        
        _   = model.run(None,{input_name:img})

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

import onnx 
import shutil
from onnxsim import simplify
import onnxruntime as ort
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from BOS_util.util import to_qdq_onnx
def main(args):
    logger_enable(args.prefix)
    logger.info('start!')
       
    ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
    processor = Processor(new_shape=(args.size, args.size))
    prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers)       
    model= onnx.load('yolov8s.onnx')
    root = f'output/{args.prefix}/'        
    os.makedirs(root,exist_ok=True)
    shutil.copyfile('_custom_config.json',root+'config.json')  
    
        
    try:    model, _ = simplify(model); print('simplify success')
    except: print('silplify failed')
    with open(root+'graph.graph', "w") as f: f.write(str(model.graph.node))      

    # session = ort.InferenceSession(model.SerializeToString(),providers=providers)   
    # eval(session, dataloader, args.size, 'default') 

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sim = QuantizationSimModel(model=model,
                            quant_scheme=QuantScheme.post_training_tf,
                            default_activation_bw=8,
                            default_param_bw=8,
                            providers=providers,
                            config_file='_custom_config.json')

    for key in sim.qc_quantize_op_dict:
        qc = sim.qc_quantize_op_dict[key]
        enabled = qc.enabled
        if not enabled: continue
        
        if not('constant' in key or 'Constant' in key or 'Squeeze' in key) : continue
        sim.qc_quantize_op_dict[key].enabled = False
    
    sim.compute_encodings(forward_pass_callback= lambda session,samples : calibrate(session, dataloader, samples), forward_pass_callback_args=4000)
    qdq_model = to_qdq_onnx(sim)
    del sim
    with open(root+'graph_qdq.graph', "w") as f: f.write(str(qdq_model.graph.node))
    onnx.save(qdq_model,root+f'{args.prefix}.onnx')
    qdq_session = ort.InferenceSession(qdq_model.SerializeToString(),providers=providers)        
    eval(qdq_session, dataloader, args.size, 'default') 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="yolo")
    parser.add_argument("--prefix", type=str, default="test1")
    parser.add_argument("--model_name", type=str, default="yolov8s.pt")
    parser.add_argument("--dataset_name", type=str, default='rafaelpadilla/coco2017')
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/COCO')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=6, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])

    parser.add_argument("--size", type=int, default=640)
    parser.add_argument('--exclude', type=str)

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   
    
    main(args)