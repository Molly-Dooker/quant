from .iou2d import bbox_overlaps
from .structures.utils import get_box_type

def bbox_overlaps_nearest_3d(bboxes1,
                             bboxes2,
                             mode='iou',
                             is_aligned=False,
                             coordinate='lidar'):
    """Calculate nearest 3D IoU.

    Note:
        This function first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.
        Ths IoU calculator :class:`BboxOverlapsNearest3D` uses this
        function to calculate IoUs of boxes.

        If ``is_aligned`` is ``False``, then it calculates the ious between
        each bbox of bboxes1 and bboxes2, otherwise the ious between each
        aligned pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry, v].
        bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry, v].
        mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).
        is_aligned (bool): Whether the calculation is aligned

    Return:
        torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
            bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
            ``False``, return shape is M.
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    # Change the bboxes to bev
    # box conversion and iou calculation in torch version on CUDA
    # is 10x faster than that in numpy version
    bboxes1_bev = bboxes1.nearest_bev
    bboxes2_bev = bboxes2.nearest_bev

    ret = bbox_overlaps(
        bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret
