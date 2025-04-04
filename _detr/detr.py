import ipdb

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrFrozenBatchNorm2d
from transformers.utils.fx import symbolic_trace
import torch
from PIL import Image
import requests
import numpy as np
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from datasets import load_dataset
from tqdm import tqdm

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

def fold_bn_into_conv(conv: nn.Conv2d, bn: DetrFrozenBatchNorm2d):
    """
    BN을 conv에 접합(fold)하여 conv의 weight와 bias를 업데이트합니다.
    즉, y = bn(conv(x)) 의 효과를 conv에 반영합니다.
    """
    bn_eps = 1e-5  # BN forward에서 사용하는 epsilon

    conv_w = conv.weight.clone().detach()
    if conv.bias is not None:
        conv_b = conv.bias.clone()
    else:
        conv_b = torch.zeros(conv_w.size(0), device=conv_w.device)
    bn_rv = bn.running_var
    bn_rm = bn.running_mean
    bn_w  = bn.weight
    bn_b  = bn.bias
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (conv_w.dim() - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    conv.weight.data.copy_(conv_w)
    if conv.bias is None:
        conv.bias = nn.Parameter(conv_b)
    else:
        conv.bias.data.copy_(conv_b)

def fold_bn_in_sequential(seq: nn.Sequential):
    """
    nn.Sequential 내부에서 인접한 Conv2d와 DetrFrozenBatchNorm2d 패턴을 찾아 folding을 수행한 후,
    BN 모듈을 nn.Identity()로 교체합니다.
    """
    new_modules = []
    i = 0
    while i < len(seq):
        m = seq[i]
        # 다음 모듈이 존재하고 현재가 Conv2d, 다음이 BN이면 folding
        if (i + 1 < len(seq) and isinstance(m, nn.Conv2d) 
                and isinstance(seq[i + 1], DetrFrozenBatchNorm2d)):
            conv = m
            bn = seq[i + 1]
            fold_bn_into_conv(conv, bn)
            new_modules.append(conv)
            new_modules.append(nn.Identity())
            i += 2
        else:
            new_modules.append(m)
            i += 1
    return nn.Sequential(*new_modules)

def recursive_fold(module: nn.Module):
    """
    재귀적으로 모듈을 순회하며,
      - nn.Sequential인 경우 내부의 conv–BN 패턴 folding 적용
      - 그 외 부모 컨테이너의 named_children()를 순회하면서 단독 BN 모듈을 nn.Identity()로 교체
    """
    # 먼저 nn.Sequential이면 folding된 새로운 Sequential로 교체합니다.
    if isinstance(module, nn.Sequential):
        new_seq = fold_bn_in_sequential(module)
        module = new_seq

    # "convolution"과 "normalization" 속성이 있다면 folding 적용
    if hasattr(module, "convolution") and hasattr(module, "normalization"):
        conv = getattr(module, "convolution")
        norm = getattr(module, "normalization")
        if isinstance(conv, nn.Conv2d) and isinstance(norm, DetrFrozenBatchNorm2d):
            fold_bn_into_conv(conv, norm)
            setattr(module, "normalization", nn.Identity())

    # 부모 컨테이너 내에서 단독 BN 모듈을 교체
    for name, child in list(module.named_children()):
        if isinstance(child, DetrFrozenBatchNorm2d):
            # 부모(module)에서 직접 교체
            setattr(module, name, nn.Identity())
        else:
            recursive_fold(child)

def fold_frozen_bn_to_identity(model: nn.Module):
    """
    모델 내의 모든 DetrFrozenBatchNorm2d에 대해,
    conv–BN 패턴이 존재하면 BN의 효과를 conv에 fold하고, 
    BN은 nn.Identity()로 교체합니다.
    """
    recursive_fold(model)



def transform(data_batch, processor):
    IMAGE = []; origin_shape = [];
    for image in data_batch['image']:
        IMAGE.append(image.convert('RGB'))
        origin_shape.append(image.size[::-1])
    inputs = processor(IMAGE, return_tensors="pt")
    inputs["image_id"] = data_batch["image_id"]
    inputs["objects"] = data_batch["objects"]
    inputs['origin_shape']=origin_shape
    inputs['image']=data_batch['image']
    return inputs


def custom_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
    image_id = [item['image_id'] for item in batch]
    objects = [item['objects'] for item in batch]
    origin_shape = [item['origin_shape'] for item in batch]
    image = [item['image'] for item in batch]
    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'objects': objects,
        'image_id': image_id,
        'objects':objects,
        'origin_shape':origin_shape,
        'image':image
    }


if __name__ == '__main__':
        # ipdb.set_trace()
        device = 'cuda:2'

        ds = load_dataset(path='rafaelpadilla/coco2017', cache_dir='/Data/Dataset/COCO', split='val')
        processor = DetrImageProcessor().from_pretrained("facebook/detr-resnet-50", revision="no_timm", size={"height": 800, "width": 800})
        prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
        dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
        # ipdb.set_trace()
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        fold_frozen_bn_to_identity(model)

        model.to(device)
        model.eval()  
        with torch.no_grad():
            for batch in tqdm(dataloader,desc='data'):
                inputs = {
                    'pixel_values': batch.get('pixel_values').to(device),
                    'pixel_mask': batch.get('pixel_mask').to(device)}
                outputs = model(**inputs)               
                origin_shape = torch.stack([torch.tensor(shape_) for shape_ in batch['origin_shape']])
                results = processor.post_process_object_detection(outputs, target_sizes=origin_shape, threshold=0.001)
                


