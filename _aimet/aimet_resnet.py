import sys
import os
import ipdb
import argparse
import itertools
from tqdm import tqdm
from util import ipdb_sys_excepthook, _ROOTDIR
from loguru import logger
ipdb_sys_excepthook()

import torch, torchvision
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers.utils.fx import symbolic_trace
import evaluate

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.v2.quantsim import QuantParams
from aimet_torch.bias_correction import correct_bias
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
# from torchvision.models import resnet18
# from aimet_torch.meta import connectedgraph

def get_label():
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    id2label = model.config.id2label
    del model
    return id2label
id2label = get_label()

def get_model(source:str,device:str):
    assert source in ['hf','torch']

    model = None
    if source=='hf':
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50",device_map=device)
        model.config.return_dict=False
        model = symbolic_trace(model,input_names=['pixel_values'])
    elif source=='torch':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
        # model = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    model = prepare_model(model)
    _ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
    return model

def get_hf_processor():
    return AutoImageProcessor.from_pretrained("microsoft/resnet-50")

def get_dataset(split='validation'):
    path="ILSVRC/imagenet-1k"
    cache_dir = '/Data/Dataset/ImageNet'
    dataset = load_dataset(path=path,cache_dir=cache_dir,split=split)
    return dataset



def collate_fn(processor, batch):
    # Convert each image to RGB and get the labels
    images = [item["image"].convert("RGB") for item in batch]
    labels = [item["label"] for item in batch]
    processed = processor(images, return_tensors="pt")['pixel_values']
    return processed, torch.tensor(labels)



def pass_calibration_data(sim_model, dataloader, device, samples=10000):  
    iter = round(samples/dataloader.batch_size)
    sim_model.eval()
    with torch.no_grad():
        for img,_ in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            img = img.to(device)
            _=sim_model(img)




def logger_enable(prefix=''):
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add("_logs/log.log",
            rotation="500 MB",
            level="INFO",
            format=LOG_FORMAT)

    # 콘솔 로거
    logger.add(sys.stdout,
            level="INFO",
            format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)

    # logger.add(f"_logs/log.log", rotation="500 MB", level="INFO")
    # logger.add(sys.stdout, level="INFO")
    # logger.add(sys.stderr, level="INFO", filter=lambda record: "hide_console" not in record["extra"])
    # logger.info(f'{model_id} start')

def evaluation(model,dataloader,device,prefix):
    metric = evaluate.load('accuracy')
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc=f"evaluate {prefix} model..."):
            img = img.to(device)
            predicted = model(img)
            if isinstance(predicted,tuple):
                predicted = predicted[0]
            predicted = predicted.argmax(-1).cpu()
            metric.add_batch(predictions=predicted,references=label)
            
    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def get_savedir(prefix:str):
    dir = _ROOTDIR+prefix+'/'
    os.makedirs(dir, exist_ok=True)
    return dir




def run(model:str, device:str, prefix:str, adaround:bool, cle:bool, evaluate_base=False, config_file='aimet_config_ch.json'):

    logger_enable(prefix)   
    save_dir = get_savedir(prefix)   
    sim      = None

    model     = get_model(model,device)
    processor = get_hf_processor()         
    ds = get_dataset()
    dataloader = DataLoader(ds, batch_size=512, collate_fn=lambda batch: collate_fn(processor, batch),shuffle=True)
    if evaluate_base:
        evaluation(model, dataloader, device, f'_base_{model}')

    dummy_input = torch.rand(1, 3, 224, 224)
    dummy_input = dummy_input.to(device)

    if cle:
        logger.info('CLE&BC start')
        equalize_model(model, input_shapes=(1, 3, 224, 224))

        bc_params = QuantParams(weight_bw=8, act_bw=8, 
                                round_mode="nearest",
                                config_file=config_file
                                # quant_scheme=QuantScheme.post_training_tf_enhanced
                                )
        correct_bias(model, bc_params,
                     data_loader=dataloader,
                     num_quant_samples=500,
                     num_bias_correct_samples=1000,                     
                     )

        if not adaround:
            sim = QuantizationSimModel(model=model,
                                    dummy_input=dummy_input,
                                    # quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    config_file=config_file)
        logger.info('CLE&BC end')

    if adaround:
        logger.info('AdaRound start')
        ada_params = AdaroundParameters(data_loader=dataloader, num_batches=4, default_num_iterations=10000)
        ada_model = Adaround.apply_adaround(model, dummy_input, ada_params,
                                            path=save_dir,
                                            filename_prefix=prefix,
                                            default_param_bw=8,
                                            # default_quant_scheme=QuantScheme.post_training_tf_enhanced,
                                            default_config_file=config_file)

        sim = QuantizationSimModel(model=ada_model,
                                dummy_input=dummy_input,
                                # quant_scheme=QuantScheme.post_training_tf_enhanced,
                                default_output_bw=8,
                                default_param_bw=8,
                                config_file=config_file)

        sim.set_and_freeze_param_encodings(encoding_path=f'{save_dir}{prefix}.encodings')
        logger.info('AdaRound end')

    if not (cle or adaround):
        sim = QuantizationSimModel(model=model,
                                dummy_input=dummy_input,
                                # quant_scheme=QuantScheme.post_training_tf_enhanced,
                                default_output_bw=8,
                                default_param_bw=8,
                                config_file=config_file)


    sim.compute_encodings(forward_pass_callback=lambda sim_model: pass_calibration_data(sim_model, dataloader, device))
    logger.info('Evaluation start')
    evaluation(sim.model,dataloader,device, prefix)
    logger.info('Evaluation end')
    dummy_input = dummy_input.cpu()
    sim.export(path=save_dir, filename_prefix=prefix, dummy_input=dummy_input)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, required=True, help='') # hf, torch
    parser.add_argument('--device', type=int, required=True, help='') # 0,1,2,3,4,5,6,7
    parser.add_argument('--prefix', type=str, required=True, help='') # e.g _ada_hf, _cle_hf
    parser.add_argument('--cle', type=bool, required=True, help='') # True, False
    parser.add_argument('--adaround', type=bool, required=True, help='') # True, False


    # 커맨드라인 인자를 파싱합니다.
    args = parser.parse_args()   


    model    = args.model
    device   = f'cuda:{args.device}'
    prefix   = args.prefix
    adaround = args.adaround
    cle      = args.cle


    run(model, device, prefix, adaround, cle)

