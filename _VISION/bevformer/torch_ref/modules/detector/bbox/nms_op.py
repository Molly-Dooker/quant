import torch

# from modules.utils import load_ext

# ext_module = load_ext(
#     'modules.detector.bbox._ext', ['nms', 'softnms', 'nms_match', 'nms_rotated'])

class NMSop(torch.autograd.Function):
    pass
    # @staticmethod
    # def forward(ctx, bboxes, scores, iou_threshold, offset, score_threshold,
    #             max_num):
    #     is_filtering_by_score = score_threshold > 0
    #     if is_filtering_by_score:
    #         valid_mask = scores > score_threshold
    #         bboxes, scores = bboxes[valid_mask], scores[valid_mask]
    #         valid_inds = torch.nonzero(
    #             valid_mask, as_tuple=False).squeeze(dim=1)

    #     inds = ext_module.nms(
    #         bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)

    #     if max_num > 0:
    #         inds = inds[:max_num]
    #     if is_filtering_by_score:
    #         inds = valid_inds[inds]
    #     return inds

    # @staticmethod
    # def symbolic(g, bboxes, scores, iou_threshold, offset, score_threshold,
    #              max_num):
    #     from ..onnx import is_custom_op_loaded
    #     has_custom_op = is_custom_op_loaded()
    #     # TensorRT nms plugin is aligned with original nms in ONNXRuntime
    #     is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
    #     if has_custom_op and (not is_trt_backend):
    #         return g.op(
    #             'mmcv::NonMaxSuppression',
    #             bboxes,
    #             scores,
    #             iou_threshold_f=float(iou_threshold),
    #             offset_i=int(offset))
    #     else:
    #         from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze
    #         from ..onnx.onnx_utils.symbolic_helper import _size_helper

    #         boxes = unsqueeze(g, bboxes, 0)
    #         scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)

    #         if max_num > 0:
    #             max_num = g.op(
    #                 'Constant',
    #                 value_t=torch.tensor(max_num, dtype=torch.long))
    #         else:
    #             dim = g.op('Constant', value_t=torch.tensor(0))
    #             max_num = _size_helper(g, bboxes, dim)
    #         max_output_per_class = max_num
    #         iou_threshold = g.op(
    #             'Constant',
    #             value_t=torch.tensor([iou_threshold], dtype=torch.float))
    #         score_threshold = g.op(
    #             'Constant',
    #             value_t=torch.tensor([score_threshold], dtype=torch.float))
    #         nms_out = g.op('NonMaxSuppression', boxes, scores,
    #                        max_output_per_class, iou_threshold,
    #                        score_threshold)
    #         return squeeze(
    #             g,
    #             select(
    #                 g, nms_out, 1,
    #                 g.op(
    #                     'Constant',
    #                     value_t=torch.tensor([2], dtype=torch.long))), 1)

