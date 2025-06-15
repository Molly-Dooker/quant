import argparse
import sys
import itertools
import math
import ipdb
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, transform
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.quantization as quant
from torch.quantization import (
    FakeQuantize,
    FakeQuantizeBase,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    HistogramObserver,
    prepare_qat,
    convert,
)
import numpy as np
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
import os

from torch.fx import symbolic_trace, GraphModule
from torch_config.myconfig import simple_qconfig_mapping, default_qconfig_mapping
# from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import (
    prepare_fx, 
    convert_fx,
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


    



def eval(model, test_loader, prefix=''):

    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]

    metric = evaluate.load("accuracy")

    for batch in tqdm(test_loader,desc='eval...'):
        data, target = batch["pixel_values"].numpy(), batch["labels"].numpy()
        output = model.run(output_names,{input_name: data})[0]
        output = output.argmax(-1)
        metric.add_batch(predictions=output,references=target)

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, dataloader, samples=1000):
    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]
    iter =  min(math.ceil(samples/dataloader.batch_size), dataloader.__len__())
    for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
        data = batch["pixel_values"].numpy()
        output = model.run(output_names,{input_name: data})[0]

def calibrate_wrapper(model, samples, dataloader):
    calibrate(model,dataloader,samples)
    
    
    


import shutil
import onnxruntime as ort
from onnxsim import simplify
import onnx
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from copy import deepcopy
from aimet_common.onnx._utils import _add_onnx_qdq_nodes
from aimet_common.defs import QuantScheme, QuantizationDataType


def _remove_final_qdq(qdq_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    생성된 QDQ 모델의 최종 출력에 불필요하게 붙은 QDQ 노드와 관련 Initializer를 제거합니다.
    :param qdq_model: QDQ 노드가 포함된 ONNX 모델
    :return: 최종 출력 QDQ가 제거된 ONNX 모델
    """
    # print("\n--- 최종 출력 QDQ 노드 및 관련 Initializer 제거 시작 ---")
    graph = qdq_model.graph
    
    # 모델의 최종 출력 텐서 이름들을 복사해서 사용 (순회 중 변경될 수 있으므로)
    original_output_names = [output.name for output in graph.output]
    
    nodes_to_remove = []
    initializers_to_remove = set()

    for output_name in original_output_names:
        # 최종 출력을 생성하는 Dequantize 노드를 찾음
        final_dq_node = next((n for n in graph.node if n.op_type == 'DequantizeLinear' and n.output[0] == output_name), None)
        
        if not final_dq_node:
            continue
            
        quantized_tensor_name = final_dq_node.input[0]
        final_q_node = next((n for n in graph.node if n.op_type == 'QuantizeLinear' and n.output[0] == quantized_tensor_name), None)
        
        if not final_q_node:
            continue

        original_tensor_name = final_q_node.input[0]
        scale_name = final_q_node.input[1]
        zp_name = final_q_node.input[2]
        
        # print(f"   - 제거 대상 QDQ 쌍 확인: ('{final_q_node.name}', '{final_dq_node.name}')")
        # print(f"   - 제거 대상 Initializer 확인: ('{scale_name}', '{zp_name}')")
        
        # 모델의 최종 출력을 QDQ 이전의 텐서로 변경
        for out_val_info in graph.output:
            if out_val_info.name == output_name:
                out_val_info.name = original_tensor_name
        
        nodes_to_remove.extend([final_q_node, final_dq_node])
        initializers_to_remove.update([scale_name, zp_name])
    
    # 실제 노드 및 Initializer 제거
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    # 제거 대상이 아닌 Initializer들만 남김
    remaining_initializers = [init for init in graph.initializer if init.name not in initializers_to_remove]
    graph.ClearField("initializer")
    graph.initializer.extend(remaining_initializers)
        
    # print("--- 노드 및 Initializer 제거 완료 ---")
    return qdq_model





def _to_onnx_qdq(sim) -> onnx.ModelProto:
    """
    Return a copy of ModelProto with all QcQuantizeOp replaced with
    onnx::QuantizeLinear and/or DequantizeLinear
    """
    onnx_opset_version = next(
        opset.version for opset in sim.model.opset_import() if opset.domain == ""
    )

    desired_onnx_opset_version = onnx_opset_version

    if onnx_opset_version < 10:
        desired_onnx_opset_version = 10

        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear are only supported in opset >= 10;"
            " got opset=%d",
            onnx_opset_version,
        )

    if onnx_opset_version < 13 and any(
        qtzr.quant_info.usePerChannelMode
        and qtzr.tensor_quantizer_params
        and qtzr.tensor_quantizer_params.channel_axis is not None
        for qtzr in sim.qc_quantize_op_dict.values()
    ):
        desired_onnx_opset_version = 13
        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear with per-channel are only supported in opset >= 13;"
            " got opset=%d",
            onnx_opset_version,
        )

    if onnx_opset_version < 21 and any(
        qtzr.quant_info.usePerChannelMode
        and qtzr.tensor_quantizer_params
        and qtzr.quant_info.blockSize > 0
        for qtzr in sim.qc_quantize_op_dict.values()
    ):
        desired_onnx_opset_version = 21
        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
            " got opset=%d",
            onnx_opset_version,
        )

    if onnx_opset_version < 21 and any(
        qtzr.data_type == QuantizationDataType.int and 8 < qtzr.bitwidth <= 16
        for qtzr in sim.qc_quantize_op_dict.values()
    ):
        desired_onnx_opset_version = 21
        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear with INT16 are only supported in opset >= 21;"
            " got opset=%d",
            onnx_opset_version,
        )

    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(sim.model.model)

    sim._overwrite_parameters(model_copy, sim._get_qdq_parameters())

    aimet_qc_quantize_nodes = [
        node
        for node in model_copy.graph.node
        if node.op_type == "QcQuantizeOp"
        and node.domain in ("aimet.customop.cpu", "aimet.customop.cuda")
    ]

    qdq_node_info = {
        "input_names": [],
        "output_names": [],
        "node_name_prefixes": [],
        "encodings": [],
    }

    for aimet_node in aimet_qc_quantize_nodes:
        qtzr = sim.qc_quantize_op_dict[aimet_node.input[0]]
        encodings = qtzr._export_2_0_0_encodings()  # pylint: disable=protected-access

        if encodings:
            # Affine quantizer
            # Replace QcQuantizeOp with onnx::QuantizeLinear and DequantizeLinear
            qdq_node_info["input_names"].append(aimet_node.input[0])
            qdq_node_info["output_names"].append(aimet_node.output[0])
            qdq_node_info["node_name_prefixes"].append(aimet_node.name)
            qdq_node_info["encodings"].append(encodings)

    graph_output_names = [out.name for out in model_copy.graph.output]
    sim.remove_quantizers(model_copy)

    if onnx_opset_version < desired_onnx_opset_version:
        model_copy = onnx.version_converter.convert_version(
            model_copy, desired_onnx_opset_version
        )

    _add_onnx_qdq_nodes(
        model_copy, **qdq_node_info, onnx_opset=desired_onnx_opset_version
    )

    # Graph output could have been renamed during self.remove_quantizers.
    # Restore the original output names
    q_input_names = set(
        node.input[0]
        for node in model_copy.graph.node
        if node.op_type == "QuantizeLinear"
    )
    dq_output_names = set(
        node.output[0]
        for node in model_copy.graph.node
        if node.op_type == "DequantizeLinear"
    )
    for out, orig_name in zip(model_copy.graph.output, graph_output_names):
        if out.name in q_input_names and orig_name in dq_output_names:
            out.name = orig_name

    # TODO: Unfortunately, this sanity check doesn't pass yet because the
    #       QcQuantizeOp nodes inserted during QuantizationSimModel.__init__
    #       aren't topologically sorted, but onnx.checker asserts topological
    #       order of all nodes. Needs to be fixed asap.
    # onnx.checker.check_model(model_copy, True)
    qdq_model = _remove_final_qdq(model_copy)
    return qdq_model


def main(args):
    logger_enable(args.prefix)
    EVAL = args.eval
    if not EVAL : 
        
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dummy_input = torch.randn(1, 3, 224, 224)           
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        filename = 'resnet18.onnx'
        model = onnx.load_model(filename)
        # eval(session,dataloader,args.prefix)
        root = f'output/{args.prefix}/'        
        os.makedirs(root,exist_ok=True)
        shutil.copyfile('_custom_config.json',root+'config.json')        
        
        with open(root+'graph.graph', "w") as f:
            f.write(str(model.graph.node))        
        try:
            model, _ = simplify(model)
        except:
            pass
            # print('ONNX Simplifier failed. Proceeding with unsimplified model')
        sim = QuantizationSimModel(model=model,
                                quant_scheme=QuantScheme.post_training_tf,
                                default_activation_bw=8,
                                default_param_bw=8,
                                providers=providers,
                                config_file='_custom_config.json')
            
        sim.compute_encodings(forward_pass_callback=lambda session,samples : calibrate_wrapper(session,samples,dataloader),
                            forward_pass_callback_args=1000)
        
        qdq_model = _to_onnx_qdq(sim)
        
        qdq_session = ort.InferenceSession(qdq_model.SerializeToString(),providers=providers)        
        eval(qdq_session,dataloader,'dd')
        
        onnx.save(qdq_model,root+'qdq.onnx')
        with open(root+'graph_qdq.graph', "w") as f:
            f.write(str(qdq_model.graph.node))
        
        





        






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="resnet")
    parser.add_argument("--prefix", type=str, default="resnet1")
    # parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=6, help="The device to use for evaluation.")
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