
from ultralytics.utils.ops import non_max_suppression, scale_boxes

def _get_nms(preds,origin_shapes,conf_thres=0.25,iou_thres=0.7,scaled_shape=(640,640)):
    preds = non_max_suppression(preds,conf_thres,iou_thres,None,False,max_det=300,nc=80,end2end=False,rotated=False)
    for pred, origin_shape in zip(preds,origin_shapes):
        pred[:, :4] = scale_boxes(scaled_shape, pred[:, :4], origin_shape)
    return preds