import sys
from optimum.quanto import (
    qfloat8,
    qint4,
    qint8,
)

from ultralytics.utils.metrics import  box_iou
from ultralytics.utils.ops import scale_boxes

import torch
import numpy as np

class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush"
            }
names_reversed = {v: k for k, v in class_names.items()}
mapper = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}


def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def label_mapper(cls):
    cls_ = [names_reversed[mapper[c]] for c in cls]
    cls_ = torch.tensor(cls_,dtype=torch.float32)
    return cls_

def bbox_mapper(bbox):
    if len(bbox)==0:
        return torch.empty((0, 4))
    bbox_ = [torch.tensor(box) for box in bbox]
    bbox_ = torch.stack(bbox_)
    bbox_[:,2:]=bbox_[:,:2]+bbox_[:,2:]
    return bbox_

def _restore_scale(pred, origin_shape, scaled_shape=(640,640)):
    predn = pred.clone()
    scale_boxes(scaled_shape, predn[:, :4], origin_shape)
    return predn

def _match_predictions(
    pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
) -> torch.Tensor:
    """
    Match predictions to ground truth objects using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape (N,).
        true_classes (torch.Tensor): Target class indices of shape (M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
    """
    iouv = torch.tensor([0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500])
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def _process_batch(detections, gt_bboxes, gt_cls):
    iou = box_iou(gt_bboxes, detections[:, :4])      
    result = _match_predictions(detections[:, 5], gt_cls, iou)
    return result


def update_stats(preds, objects, origin_shapes, stats, device, size=640):
    for si, pred in enumerate(preds):
        npr = len(pred)
        stat = dict(
            conf=torch.zeros(0, device=device),
            pred_cls=torch.zeros(0, device=device),
            tp=torch.zeros(npr, 10, dtype=torch.bool, device=device),
        )            
        origin_shape = origin_shapes[si]
        info = objects[si]
        cls  = label_mapper(info['label']).to(device)
        bbox = bbox_mapper(info['bbox']).to(device)
        nl = len(cls)
        stat["target_cls"] = cls
        stat["target_img"] = cls.unique()
        if npr == 0:
            if nl:
                for k in stats.keys():
                    stats[k].append(stat[k])
            continue        
        predn=_restore_scale(pred, origin_shape,(size,size))
        stat["conf"] = predn[:, 4]
        stat["pred_cls"] = predn[:, 5]
        if nl:
            stat["tp"] = _process_batch(predn, bbox, cls)
        for k in stats.keys():
            stats[k].append(stat[k])
    return stats

# def get_preds(nms):
#     preds=[]
#     for nms_ in nms:
#         pred = dict()        
#         nms_ = nms_.cpu()
#         pred['boxes']=nms_[:,:4]
#         pred['scores']=nms_[:,4]
#         pred['labels']=nms_[:,-1].to(torch.int32)
#         preds.append(pred)
#     return preds

# def get_targets(objects):
#     targets=[]
#     for object in objects:
#         target = dict()
#         bboxes = object['bbox']
#         boxes = []
#         for bbox in bboxes:
#             xyxy = torch.tensor([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
#             boxes.append(xyxy)
#         if len(boxes)==0: 
#             boxes = torch.tensor([])
#             labels= torch.tensor([],dtype=torch.int32)
#         else:
#             boxes = torch.stack(boxes)
#             labels = torch.tensor(object['label'],dtype=torch.int32)-1
#         target['boxes']=boxes;
#         target['labels']=labels;
#         targets.append(target)
#     return targets;