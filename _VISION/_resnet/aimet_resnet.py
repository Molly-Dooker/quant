import argparse
import sys
import json
import itertools
import math
import ipdb
import os
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, transform
import torchvision
import torch
import evaluate
from accelerate import init_empty_weights
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    AutoImageProcessor,
)
from optimum.quanto import (
    QTensor,
    freeze,
    qfloat8,
    qint4,
    qint8,
    quantization_map
)
from _quanto import _quantize, _requantize, _Calibration
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

def eval(model, test_loader, prefix=''):
    metric = evaluate.load("accuracy")
    input_name = model.get_inputs()[0].name
    for batch in tqdm(test_loader,desc='eval...'):
        data, target = batch["pixel_values"], batch["labels"]
        data = data.numpy()
        output = model.run(None,{input_name:data})[0]
        output =  output.argmax(-1)
        metric.add_batch(predictions=output,references=target)
    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')




def calibrate(model, dataloader, num=10000):
    input_name = model.get_inputs()[0].name
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
        # ipdb.set_trace()
        data = batch["pixel_values"].numpy()
        _    = model.run(None,{input_name:data})[0]

# def calibrate_wrapper(model, samples, dataloader):
#     calibrate(model, dataloader, samples)

import onnx 
import shutil
from onnxsim import simplify
import onnxruntime as ort
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
import BOS_config.aimet_util as aimet_util
def main(args):
    logger_enable(args.prefix)
    EVAL = args.eval
    if not EVAL : 
        # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        # ipdb.set_trace()
        # # model2 = ORTModelForImageClassification("microsoft/resnet-50",export=True)
        # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()       



        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        data_loader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # print("Model before quantization...")
        
        root = f'output/{args.prefix}/'        
        os.makedirs(root,exist_ok=True)
        shutil.copyfile('_custom_config.json',root+'config.json')       

        model = onnx.load('resnet50.onnx')
        dummy_input = torch.randn(1, 3, 224, 224)
        

            
        try:    model, _ = simplify(model); print('simplify success')
        except: print('silplify failed')
        with open(root+'graph.graph', "w") as f: f.write(str(model.graph.node))   

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # eval(session, dataloader, 'default') 80.14

        sim = QuantizationSimModel(model=model,
                                quant_scheme=QuantScheme.post_training_tf,
                                default_activation_bw=8,
                                default_param_bw=8,
                                providers=providers,
                                config_file='_custom_config.json')
        sim.qc_quantize_op_dict['classifier.1.weight'].enabled=False
            



        sim.compute_encodings(forward_pass_callback= lambda session,samples : calibrate(session, data_loader, samples), forward_pass_callback_args=2000)

        with TempLoggerPatch(aimet_util, logger):
            qdq_model = aimet_util._to_onnx_qdq(sim)
        del sim
        with open(root+'graph_qdq.graph', "w") as f:
            f.write(str(qdq_model.graph.node))
        onnx.save(qdq_model,root+'qdq.onnx')
        qdq_session = ort.InferenceSession(qdq_model.SerializeToString(),providers=providers)        
        eval(qdq_session, data_loader, 'quantized')  #79.39




    # if EVAL:
    #     processor = ViTImageProcessor.from_pretrained(args.model_name)
    #     ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
    #     prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    #     dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #     state_dict = load_file(f'{args.saveroot}/{args.prefix}.safetensors')
    #     with open(f'{args.saveroot}/{args.prefix}.json', 'r') as f:
    #         qmap = json.load(f)
    #     config = ViTConfig.from_pretrained(args.model_name)
    #     with init_empty_weights():
    #         model = ViTForImageClassification.from_pretrained(args.model_name, config=config)
    #     _requantize(model, state_dict, qmap, args.device)
    #     freeze(model)
    #     eval(model, args.device, dataloader,'reloaded')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="resnet")
    parser.add_argument("--prefix", type=str, default="resnet1")
    # parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=7, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    parser.add_argument('--exclude', type=str)

    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   

    main(args)