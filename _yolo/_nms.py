
from ultralytics.utils.ops import non_max_suppression, scale_boxes

def _get_nms(preds,origin_shapes,conf_thres=0.25,iou_thres=0.7,scaled_shape=(640,640),labels=[],nc=80,multi_label=True,agnostic=False,max_det=300,end2end=False,rotated=False):
    preds = non_max_suppression(preds,conf_thres,iou_thres,labels=labels,nc=nc,multi_label=multi_label,agnostic=agnostic,max_det=max_det,end2end=end2end,rotated=rotated)
    for pred, origin_shape in zip(preds,origin_shapes):
        pred[:, :4] = scale_boxes(scaled_shape, pred[:, :4], origin_shape)
    return preds