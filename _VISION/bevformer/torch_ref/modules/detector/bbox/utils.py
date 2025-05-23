# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import torch
from enum import IntEnum, unique

import warnings
from copy import copy
from bevformer.torch_ref.modules import utils
from .nms_op import NMSop
from bevformer.torch_ref.modules.utils import load_ext

ext_module = 0 # load_ext(
#     'modules.detector.bbox._ext', ['nms', 'softnms', 'nms_match', 'nms_rotated'])


# STILL MMDET3D
# from mmdet3d.core.bbox import BaseInstance3DBoxes
# from mmdet3d.core.bbox import CameraInstance3DBoxes
# from mmdet3d.core.bbox import DepthInstance3DBoxes
# from mmdet3d.core.bbox import LiDARInstance3DBoxes

# from .base_box3d import BaseInstance3DBoxes
# from .cam_box3d import CameraInstance3DBoxes
# from .depth_box3d import DepthInstance3DBoxes
# from .lidar_box3d import LiDARInstance3DBoxes

def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    if torch.__version__ == 'parrots':
        indata_list = [boxes, scores]
        indata_dict = {
            'iou_threshold': float(iou_threshold),
            'offset': int(offset)
        }
        inds = ext_module.nms(*indata_list, **indata_dict)
    else:
        inds = NMSop.apply(boxes, scores, iou_threshold, offset,
                           score_threshold, max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds

def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

EPS = 1e-2


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = utils.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(utils.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = utils.color_val_matplotlib(bbox_color)
    text_color = utils.color_val_matplotlib(text_color)

    img = utils.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = utils.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        utils.imwrite(img, out_file)

    plt.close()

    return img

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # -1 indexing works abnormal in TensorRT
        # This assumes `dets` has 5 dimensions where
        # the last dimension is score.
        # TODO: more elegant way to handle the dimension issue.
        # Some type of nms would reweight the score, such as SoftNMS
        scores = dets[:, 4]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    return torch.cat([boxes, scores[:, None]], -1), keep


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)



def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """

    cfg = copy.deepcopy(cfg)

    # deprecate arguments warning
    if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
        warnings.warn(
            'In rpn_proposal or test_cfg, '
            'nms_thr has been moved to a dict named nms as '
            'iou_threshold, max_num has been renamed as max_per_img, '
            'name of original arguments and the way to specify '
            'iou_threshold of NMS will be deprecated.')
    if 'nms' not in cfg:
        cfg.nms = dict(dict(type='nms', iou_threshold=cfg.nms_thr))
    if 'max_num' in cfg:
        if 'max_per_img' in cfg:
            assert cfg.max_num == cfg.max_per_img, f'You set max_num and ' \
                f'max_per_img at the same time, but get {cfg.max_num} ' \
                f'and {cfg.max_per_img} respectively' \
                f'Please delete max_num which will be deprecated.'
        else:
            cfg.max_per_img = cfg.max_num
    if 'nms_thr' in cfg:
        assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set ' \
            f'iou_threshold in nms and ' \
            f'nms_thr at the same time, but get ' \
            f'{cfg.nms.iou_threshold} and {cfg.nms_thr}' \
            f' respectively. Please delete the nms_thr ' \
            f'which will be deprecated.'

    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals[:, :4].contiguous(),
                              aug_proposals[:, -1].contiguous(),
                              cfg.nms.iou_threshold)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(cfg.max_per_img, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]


@unique
class Box3DMode(IntEnum):
    r"""Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is [0.5, 1.0, 0.5],
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth mode:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(box, src, dst, rt_mat=None):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.ndarray |
                torch.Tensor | BaseInstance3DBoxes):
                Can be a k-tuple, k-list or an Nxk array/tensor, where k = 7.
            src (:obj:`Box3DMode`): The src Box mode.
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            (tuple | list | np.ndarray | torch.Tensor | BaseInstance3DBoxes): \
                The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'Box3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([y_size, z_size, x_size], dim=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([z_size, x_size, y_size], dim=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([y_size, x_size, z_size], dim=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([y_size, x_size, z_size], dim=-1)
        else:
            raise NotImplementedError(
                f'Conversion from Box3DMode {src} to {dst} '
                'is not supported yet')

        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [arr[:, :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[:, :3] @ rt_mat.t()

        remains = arr[..., 6:]
        arr = torch.cat([xyz[:, :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type}'
                    ' is not supported yet')
            return target_type(
                arr, box_dim=arr.size(-1), with_yaw=box.with_yaw)
        else:
            return arr
