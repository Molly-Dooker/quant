import ttnn
from bos_metal import dirs, torch

# Define tensor store folder and format
tensor_store_dir = dirs.models_gendir.make("bevformer/tensors")
tensor_ext = ".pt"
    
def get_list_shape(tensor):
    return [i for i in tensor.shape]

def add_dim(tensor, dim):
    tensor_shape = get_list_shape(tensor)
    for i in range(len(dim)):
        if dim[i]==0:
            dim[i] = tensor_shape.pop(0)
    return ttnn.reshape(tensor, dim)

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