import argparse
import sys
import json
import itertools
import math
import ipdb
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, _transform

import torch
import evaluate
from accelerate import init_empty_weights
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
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




def eval(model, device, test_loader, prefix=''):
    model.to(device)
    model.eval()
    metric = evaluate.load("accuracy")
    with torch.no_grad():
        for batch in tqdm(test_loader,desc='eval...'):
            data, target = batch["pixel_values"], batch["labels"]
            data= data.to(device)
            output = model(data)
            output = output.argmax(-1).cpu()
            metric.add_batch(predictions=output,references=target)

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            # ipdb.set_trace()
            data = batch["pixel_values"].to(device)
            _ = model(data)

import torchvision
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from BOS_util import to_qdq_torch,save_graph
import shutil
import onnx
import onnxruntime as ort
def main(args):
    logger_enable(args.prefix)
    config_path = '_default_config.json'
    # processor = ViTImageProcessor.from_pretrained(args.model_name)
    model2 = ViTForImageClassification.from_pretrained(args.model_name)        
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).eval()
    ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
    prepared_ds = ds.with_transform(lambda batch: _transform(batch, torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()))    
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # eval(model, args.device, dataloader,'default') 81.88
    dummy_input = torch.randn(1, 3, 224, 224)    

    model = prepare_model(model)
    fold_all_batch_norms(model, dummy_input.shape, dummy_input=dummy_input)
    model.to(args.device); dummy_input = dummy_input.to(args.device)
    # jit_model = torch.jit.trace(model,dummy_input)
    # mha = model.encoder.layers.encoder_layer_0.self_attention
    # torch.nn.modules.activation.MultiheadAttention
    ipdb.set_trace()
    sim = QuantizationSimModel(model=model,
                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                            dummy_input=dummy_input,
                            default_output_bw=8,
                            default_param_bw=8,
                            config_file=config_path)

    exclude = ['vit.encoder.layer.5.output.dense',
                'vit.encoder.layer.9.attention.attention.query']
    if args.exclude is not None:
        exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
        if args.exclude=='': exclude = []
    logger.info(f'exclude : {exclude}')   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ViT")
    parser.add_argument("--prefix", type=str, default="ViT")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=4, help="The device to use for evaluation.")
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