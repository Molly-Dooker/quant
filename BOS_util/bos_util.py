import contextlib
import copy
import io
import os
import tempfile
import traceback
from typing import Any, Mapping, Tuple, Union
from loguru import logger
import onnx
import torch
from torch.onnx import _constants

from aimet_common.onnx._utils import _add_onnx_qdq_nodes, _is_grid_preserving_op

# from aimet_torch.nn import QuantizationMixin
# from aimet_torch.quantization import DequantizedTensor
# from aimet_torch.quantization.base import EncodingBase
# from aimet_torch.quantization.affine import AffineQuantizerBase, GroupedBlockQuantizeDequantize
# from aimet_torch.quantization.float import FloatQuantizeDequantize
# from aimet_torch.quantsim import QuantizationSimModel
# from aimet_torch.v2.experimental import onnx as _onnx
from aimet_common.defs import QuantizationDataType

_TORCH_DEFAULT_OPSET = _constants.ONNX_DEFAULT_OPSET
_TORCH_MIN_OPSET = _constants.ONNX_MIN_OPSET
_TORCH_MAX_OPSET = _constants.ONNX_MAX_OPSET

# Allow at least up to opset 21 to enable [u]int16 QDQ export
_AIMET_MAX_OPSET = max(_TORCH_MAX_OPSET, 21)


def to_onnx_qdq(engine) -> onnx.ModelProto:
    """
    Return a copy of ModelProto with all QcQuantizeOp replaced with
    onnx::QuantizeLinear and/or DequantizeLinear
    """
    try:
        invalid_bitwidth = next(
            qtzr.bitwidth
            for qtzr in engine.qc_quantize_op_dict.values()
            if qtzr.data_type == QuantizationDataType.int
            and qtzr.bitwidth not in (4, 8, 16, 32)
        )
    except StopIteration:
        invalid_bitwidth = None

    if invalid_bitwidth is not None:
        raise RuntimeError(
            f"Invalid bitwidth {invalid_bitwidth};"
            " expected standard ONNX integer data types such as [U]INT{4, 8, 16, 32}"
        )

    onnx_opset_version = next(
        opset.version for opset in engine.model.opset_import() if opset.domain == ""
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
        for qtzr in engine.qc_quantize_op_dict.values()
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
        for qtzr in engine.qc_quantize_op_dict.values()
    ):
        desired_onnx_opset_version = 21
        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
            " got opset=%d",
            onnx_opset_version,
        )

    if onnx_opset_version < 21 and any(
        qtzr.data_type == QuantizationDataType.int and qtzr.bitwidth not in (8, 32)
        for qtzr in engine.qc_quantize_op_dict.values()
    ):
        desired_onnx_opset_version = 21
        logger.info(
            "onnx::QuantizeLinear and DequantizeLinear with INT4/INT16 are only supported in opset >= 21;"
            " got opset=%d",
            onnx_opset_version,
        )

    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(engine.model.model)

    engine._overwrite_parameters(model_copy, engine._get_qdq_parameters())

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
        qtzr = engine.qc_quantize_op_dict[aimet_node.input[0]]
        encodings = qtzr._export_2_0_0_encodings()  # pylint: disable=protected-access

        if encodings:
            # Affine quantizer
            # Replace QcQuantizeOp with onnx::QuantizeLinear and DequantizeLinear
            qdq_node_info["input_names"].append(aimet_node.input[0])
            qdq_node_info["output_names"].append(aimet_node.output[0])
            qdq_node_info["node_name_prefixes"].append(aimet_node.name)
            qdq_node_info["encodings"].append(encodings)

    graph_output_names = [out.name for out in model_copy.graph.output]
    engine.remove_quantizers(model_copy)

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
    
def save_graph(model, path):
    with open(path, "w") as f: f.write(str(model.graph.node))
    
    
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