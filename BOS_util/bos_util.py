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


# def to_qdq_torch(
#     model: Union[torch.nn.Module, QuantizationSimModel],
#     dummy_input: Union[Tuple[Any, ...], torch.Tensor],
#     *,
#     export_int32_bias: bool = False,
#     **kwargs,
# ):
#     """
#     Export :class:`QuantizationSimModel` to onnx model with
#     onnx `QuantizeLinear`_ and `DequantizeLinear`_ embedded in the graph.

#     This function takes set of same arguments as `torch.onnx.export()`_

#     Args:
#         model: The model to be exported
#         args: Same as `torch.onnx.export()`
#         f: Same as `torch.onnx.export()`
#         export_int32_bias (bool, optional):
#             If true, generate and export int32 bias encoding on the fly (default: `True`)
#         **kwargs: Same as `torch.onnx.export()`


#     .. note::
#         Unlike `torch.onnx.export()`, this function allows up to opset 21.
#         to support 4/16-bit quantization only available in opset 21.
#         However, exporting to opset 21 is a beta feature and not fully stable yet.
#         For robustness, opset 20 or lower is recommended whenever possible.

#     .. note::
#         Dynamo-based export (`dynamo=True`) is not supported yet

#     .. _torch.onnx.export(): https://docs.pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
#     .. _QuantizeLinear: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
#     .. _DequantizeLinear: https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

#     Examples:

#         >>> aimet_torch.onnx.export(sim.model, x, f="model.onnx",
#         ...                         input_names=["input"], output_names=["output"],
#         ...                         opset_version=21, export_int32_bias=True)
#         ...
#         >>> import onnxruntime as ort
#         >>> options = ort.SessionOptions()
#         >>> options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
#         >>> sess = ort.InferenceSession("model.onnx", sess_options=options)
#         >>> onnx_output, = sess.run(None, {"input": x.detach().numpy()})
#         >>> torch.nn.functional.cosine_similarity(torch.from_numpy(onnx_output), sim.model(x))
#         tensor([1.0000, 0.9999, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
#                grad_fn=<AliasBackward0>)
#     """
#     if isinstance(model, QuantizationSimModel):
#         model = model.model

#     if not isinstance(model, torch.nn.Module):
#         raise RuntimeError(
#             f"aimet_torch.export only supports torch.nn.Module or QuantizationSimModel; got {type(model)}"
#         )
#         logger.error(f"aimet_torch.export only supports torch.nn.Module or QuantizationSimModel; got {type(model)}")

#     _check_opset_version(kwargs)
#     _check_unsupported_args(kwargs)
#     _check_non_standard_quantizer(model)

#     target_version = kwargs.pop("opset_version", _TORCH_DEFAULT_OPSET)
#     kwargs["opset_version"] = min(target_version, _TORCH_MAX_OPSET)

#     _assert_minimum_required_opset(model, target_version)

#     with contextlib.ExitStack() as stack:
#         # Unfold all param quantizers to incorporate QuantizeLinear/DequantizeLinear
#         # of those parameters in tracing time
#         stack.enter_context(_temporarily_unfold_param_quantizers(model))

#         if export_int32_bias:
#             # Temoprarily instantiate int32 bias quantizers
#             stack.enter_context(
#                 _concretize_int32_bias_quantizers(model, dummy_input, kwargs.get("kwargs"))
#             )

#         # Export quantize-dequantized weight
#         # pylint: disable=protected-access
#         stack.enter_context(QuantizationSimModel._apply_qdq_to_model_parameters(model))

#         # Remove [b]float16 quantizers
#         stack.enter_context(_remove_fp16_quantizers(model))

#         onnx_model, tensor_to_encoding_map = _to_onnx(model, dummy_input, **kwargs)

#     if _TORCH_MAX_OPSET < target_version:
#         try:
#             onnx_model = onnx.version_converter.convert_version(
#                 onnx_model, target_version
#             )
#         except Exception as e:  # pylint: disable=broad-exception-caught
#             f = io.StringIO()
#             traceback.print_exc(file=f)
#             reason = _why_do_i_need_opset21(model)

#             if reason:
#                 detail = (
#                     f"torch.onnx.export only supports opset<={_TORCH_MAX_OPSET}, "
#                     f"but onnx::QuantizeLinear requires opset>={target_version} for {reason}. "
#                     "As a workaround, we tried to torch.onnx.export your model "
#                     f"with opset={_TORCH_MAX_OPSET} and convert the onnx model to {target_version}, "
#                     "but failed with the following error:\n\n"
#                 )
#             else:
#                 detail = "\n\n"

#             msg = (
#                 f"Failed to convert onnx model to {target_version} due to {type(e).__name__}. {detail}"
#                 "==============================================================\n"
#                 f"{f.getvalue()}"
#                 "==============================================================\n\n"
#             )
#             logger.error(msg)
#             raise RuntimeError(msg) from e
#     # onnx.save(onnx_qdq_model, f)
    
#     _qdq_model = _to_onnx_qdq(onnx_model, tensor_to_encoding_map)
#     qdq_model = _remove_final_qdq(_qdq_model)

#     # dynamic inputsize
#     for i,input in enumerate(qdq_model.graph.input):
#         input.type.tensor_type.shape.dim[0].ClearField("dim_value")
#         input.type.tensor_type.shape.dim[0].dim_param = 'batch_size' 

#     for i,output in enumerate(qdq_model.graph.output):
#         output.type.tensor_type.shape.dim[0].ClearField("dim_value")
#         output.type.tensor_type.shape.dim[0].dim_param = 'batch_size' 
    
#     return qdq_model

# def _why_do_i_need_opset21(model: torch.nn.Module) -> str:
#     int4 = False
#     int16 = False
#     bq = False

#     for qtzr in model.modules():
#         if not isinstance(qtzr, AffineQuantizerBase):
#             continue

#         if qtzr.block_size is not None:
#             bq = True

#         if qtzr.bitwidth == 4:
#             int4 = True

#         if qtzr.bitwidth == 16:
#             int16 = True

#     reasons = []

#     if int4 or int16:
#         reasons.append("int4/int16 quantization")

#     if bq:
#         reasons.append("blockwise quantization")

#     if not reasons:
#         return ""  # This should never happen

#     if len(reasons) == 1:
#         return reasons[0]

#     return f"{reasons[0]} and {reasons[1]}"


# def _assert_minimum_required_opset(model: torch.nn.Module, target_opset: int):
#     if target_opset < 21 and any(
#         qtzr.block_size is not None
#         for qtzr in model.modules()
#         if isinstance(qtzr, AffineQuantizerBase)
#     ):
#         raise RuntimeError(
#             "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
#             f" got opset={target_opset}"
#         )

#     if target_opset < 21 and any(
#         qtzr.bitwidth in (4, 16)
#         for qtzr in model.modules()
#         if isinstance(qtzr, AffineQuantizerBase)
#     ):
#         raise RuntimeError(
#             "onnx::QuantizeLinear and DequantizeLinear with INT4/INT16 are only supported in opset >= 21;"
#             f" got opset={target_opset}"
#         )

#     if target_opset < 13 and any(
#         tuple(qtzr.shape)
#         for qtzr in model.modules()
#         if isinstance(qtzr, AffineQuantizerBase)
#     ):
#         raise RuntimeError(
#             "onnx::QuantizeLinear and DequantizeLinear with per-channel are only supported in opset >= 13;"
#             f" got opset={target_opset}"
#         )

#     if target_opset < 10:
#         raise RuntimeError(
#             "onnx::QuantizeLinear and DequantizeLinear are only supported in opset >= 10;"
#             f" got opset={target_opset}"
#         )


# def _check_opset_version(kwargs):
#     opset_version = kwargs.get("opset_version", _TORCH_DEFAULT_OPSET)

#     if not (_TORCH_MIN_OPSET <= opset_version <= _AIMET_MAX_OPSET):
#         raise ValueError(f"Unsupported ONNX opset version: {opset_version}")


# def _check_unsupported_args(kwargs):
#     export_params = kwargs.get("export_params", True)

#     if not export_params:
#         raise NotImplementedError("export_params=False is not supported yet")

#     keep_initializers_as_inputs = kwargs.get("keep_initializers_as_inputs", False)

#     if keep_initializers_as_inputs:
#         raise NotImplementedError(
#             "keep_initializers_as_inputs=True is not supported yet"
#         )

#     dynamo = kwargs.get("dynamo", False)

#     if dynamo:
#         raise NotImplementedError("dynamo=True is not supported yet")

#     do_constant_folding = kwargs.get("do_constant_folding", True)

#     if not do_constant_folding:
#         raise NotImplementedError("do_constant_folding=False is not supported yet")

#     export_modules_as_functions = kwargs.get("export_modules_as_functions", False)

#     if export_modules_as_functions:
#         raise RuntimeError("export_modules_as_functions=True is not supported")

#     operator_export_type = kwargs.get(
#         "operator_export_type", torch.onnx.OperatorExportTypes.ONNX
#     )

#     if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN:
#         raise RuntimeError(
#             "operator_export_type=OperatorExportTypes.ONNX_ATEN is not supported"
#         )


# def _check_non_standard_quantizer(model: torch.nn.Module):
#     for name, qtzr in model.named_modules():
#         if not isinstance(qtzr, AffineQuantizerBase):
#             continue

#         if isinstance(qtzr, GroupedBlockQuantizeDequantize):
#             raise NotImplementedError(
#                 "torch.onnx.exoprt doesn't support GroupedBlockQuantizeDequantize (a.k.a LPBQ) yet; "
#                 f"got '{name}' of type GroupedBlockQuantizeDequantize"
#             )

#         if qtzr.bitwidth not in (4, 8, 16, 32):
#             raise RuntimeError(
#                 "torch.onnx.exoprt only supports 4/8/16/32-bit integers; "
#                 f"got '{name}' with bitwidth={qtzr.bitwidth}"
#             )


# def _to_onnx(
#     model: torch.nn.Module, args: Union[Tuple[Any, ...], torch.Tensor], **kwargs
# ):
#     _check_float16_quantizers(model)

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_onnx_path = os.path.join(tmp_dir, "quantized_model.onnx")
#         _onnx.export(model, args, tmp_onnx_path, **kwargs)
#         onnx_model = onnx.load(tmp_onnx_path)

#         param_names = {
#             f"{layer_name}.{param_name}"
#             for layer_name, layer in model.named_modules()
#             if isinstance(layer, QuantizationMixin)
#             for param_name, quantizer in layer.param_quantizers.items()
#             if quantizer
#         }

#     tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]]
#     tensor_to_encoding_map = {
#         name: (encoding, name in param_names)
#         for name, encoding in _onnx.remove_quantization_nodes_from_onnx_graph(
#             onnx_model
#         ).items()
#     }
#     tensor_to_encoding_map |= _derive_data_movement_op_output_encoding(
#         onnx_model, tensor_to_encoding_map
#     )
#     return onnx_model, tensor_to_encoding_map


# def _derive_data_movement_op_output_encoding(
#     model: onnx.ModelProto,
#     tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
# ) -> Mapping[str, Tuple[EncodingBase, bool]]:
#     data_movement_ops = [
#         node for node in model.graph.node if _is_grid_preserving_op(node.op_type)
#     ]

#     output_encodings = {}

#     for node in data_movement_ops:
#         input_name = node.input[0]
#         output_name = node.output[0]
#         inp_encoding, _ = tensor_to_encoding_map.get(input_name, (None, None))

#         if not inp_encoding:
#             inp_encoding, _ = output_encodings.get(input_name, (None, None))

#         if not inp_encoding:
#             # No input encoding to inherit; skip
#             continue

#         if output_name in tensor_to_encoding_map:
#             # Output encoding already exists; skip
#             continue

#         output_encodings[output_name] = (copy.deepcopy(inp_encoding), False)

#     return output_encodings


# @contextlib.contextmanager
# def _concretize_int32_bias_quantizers(model, args, kwargs=None):
#     if not isinstance(args, (tuple, list)):
#         args = (args,)

#     kwargs = kwargs or {}

#     handles = []
#     orig_bias_quantizers = {
#         qmodule: qmodule.param_quantizers["bias"]
#         for qmodule in model.modules()
#         if isinstance(qmodule, QuantizationMixin)
#         and "bias" in qmodule.param_quantizers
#         and qmodule.bias is not None
#     }

#     try:
#         for qmodule, qtzr in orig_bias_quantizers.items():
#             if qtzr is not None:
#                 # Bias quantizer already exists.
#                 # This means the user created bias quantizer by him/herself
#                 # In this case, we honor the custom bias quantizer defined by the user
#                 continue

#             if "weight" in qmodule.param_quantizers and isinstance(
#                 qmodule.param_quantizers["weight"], AffineQuantizerBase
#             ):
#                 # pylint: disable=protected-access
#                 handle = qmodule.register_forward_hook(
#                     type(qmodule)._create_int32_bias_quantizer
#                 )
#                 handles.append(handle)
#         try:
#             model(*args, **kwargs)
#         finally:
#             for handle in handles:
#                 handle.remove()
#         yield
#     finally:
#         for qmodule, qtzr in orig_bias_quantizers.items():
#             qmodule.param_quantizers["bias"] = qtzr


# @contextlib.contextmanager
# def _temporarily_unfold_param_quantizers(model: torch.nn.Module):
#     # pylint: disable=protected-access
#     """
#     Temporarily re-instantiate param quantizers for ease of export
#     """
#     modules_with_folded_parameters = [
#         qmodule
#         for qmodule in model.modules()
#         if isinstance(qmodule, QuantizationMixin)
#         and any(isinstance(param, DequantizedTensor) for param in qmodule.parameters())
#     ]

#     try:
#         for qmodule in modules_with_folded_parameters:
#             qmodule._unfold_param_quantizers()
#         yield
#     finally:
#         for qmodule in modules_with_folded_parameters:
#             qmodule._fold_param_quantizers()


# @contextlib.contextmanager
# def _remove_fp16_quantizers(model: torch.nn.Module):
#     """
#     Temporarily remove [b]float16 quantizers for sim.onnx.export,
#     as sim.onnx.export does NOT support exporting [b]float16 quantizers.
#     """
#     original_containers = {}

#     try:
#         for qmodule in model.modules():
#             if not isinstance(qmodule, QuantizationMixin):
#                 continue

#             for name, qtzr in qmodule.param_quantizers.items():
#                 if isinstance(qtzr, FloatQuantizeDequantize) and (
#                     qtzr.is_float16() or qtzr.is_bfloat16()
#                 ):
#                     original_containers[(qmodule.param_quantizers, name)] = qtzr
#                     qmodule.param_quantizers[name] = None

#             for i, qtzr in enumerate(qmodule.input_quantizers):
#                 if isinstance(qtzr, FloatQuantizeDequantize) and (
#                     qtzr.is_float16() or qtzr.is_bfloat16()
#                 ):
#                     original_containers[(qmodule.input_quantizers, i)] = qtzr
#                     qmodule.input_quantizers[i] = None

#             for i, qtzr in enumerate(qmodule.output_quantizers):
#                 if isinstance(qtzr, FloatQuantizeDequantize) and (
#                     qtzr.is_float16() or qtzr.is_bfloat16()
#                 ):
#                     original_containers[(qmodule.output_quantizers, i)] = qtzr
#                     qmodule.output_quantizers[i] = None

#         yield

#     finally:
#         for (container, key), qtzr in original_containers.items():
#             container[key] = qtzr


# def _to_onnx_qdq(
#     onnx_model: onnx.ModelProto,
#     tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
# ) -> onnx.ModelProto:
#     qnn_encodings = {
#         name: encoding.to_qnn_encoding_dict("2.0.0")
#         for name, (encoding, _) in tensor_to_encoding_map.items()
#     }
#     qnn_encodings = {
#         name: encoding for name, encoding in qnn_encodings.items() if encoding
#     }

#     qdq_tensor_names = {
#         fp_tensor_name: f"{fp_tensor_name}_qdq" for fp_tensor_name in qnn_encodings
#     }

#     onnx_opset_version = next(
#         opset.version for opset in onnx_model.opset_import if opset.domain == ""
#     )

#     # Add onnx QDQ nodes in batch
#     _add_onnx_qdq_nodes(
#         onnx_model,
#         input_names=qnn_encodings.keys(),
#         output_names=qdq_tensor_names.values(),
#         node_name_prefixes=qnn_encodings.keys(),
#         encodings=qnn_encodings.values(),
#         onnx_opset=onnx_opset_version,
#     )

#     # Restore model output names from "{output}_qdq" to "{output}"
#     _restore_model_output_names(onnx_model, qdq_tensor_names)

#     return onnx_model


# def _check_float16_quantizers(module: torch.nn.Module):
#     for qtzr in module.modules():
#         if isinstance(qtzr, FloatQuantizeDequantize):
#             if not qtzr.is_float16() and not qtzr.is_bfloat16():
#                 msg = " ".join(
#                     [
#                         "sim.onnx.export doesn't support exporting floating point encodings",
#                         f"except [b]float16. Got {qtzr.bitwidth}-bit float encoding",
#                     ]
#                 )
#                 raise RuntimeError(msg)


# def _rename_inputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
#     for node in onnx_model.graph.node:
#         for i, old_name in enumerate(node.input):
#             new_name = new_names.get(old_name, None)
#             if new_name is not None:
#                 node.input[i] = new_name


# def _rename_outputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
#     for node in onnx_model.graph.node:
#         for i, old_name in enumerate(node.output):
#             new_name = new_names.get(old_name, None)
#             if new_name is not None:
#                 node.output[i] = new_name

# def _restore_model_output_names(
#     onnx_model: onnx.ModelProto, new_names: Mapping[str, str]
# ):
#     """
#     Rename model outputs. Assuming "output" is the model output,

#     before:
#         Softmax ----> output -------> QDQ -------> output_qdq

#     after:
#         Softmax ----> output__ -----> QDQ -------> output
#     """
#     _new_names = {
#         output.name: f"{output.name}__"
#         for output in onnx_model.graph.output
#         if output.name in new_names
#     }
#     _rename_inputs(onnx_model, _new_names)

#     _new_names.update(
#         {
#             new_names[output.name]: output.name
#             for output in onnx_model.graph.output
#             if output.name in new_names
#         }
#     )
#     _rename_outputs(onnx_model, _new_names)
    
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