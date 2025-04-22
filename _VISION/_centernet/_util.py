import sys
from optimum.quanto import (
    qfloat8,
    qint4,
    qint8,
)
from transformers.models.detr.modeling_detr import DetrFrozenBatchNorm2d
import ipdb
from typing import Any, List, Mapping, Tuple, Union
import torch
import torch.nn as nn
import cv2
import numpy as np
from _centernet import Config, CenterNet
from utils.functions import ctdet_decode
import cv2
import sys
from PIL import Image
from _centernet import CenterNet, Config, DeformConv, DeformConv2, deformconv2d
from optimum.quanto.quantize import set_module_by_name
from typing import Any, Dict, List, Optional, Union
import torch
from torch.nn import functional as F, init
from optimum.quanto.nn import QModuleMixin, quantize_module
from optimum.quanto.tensor import Optimizer, qtype
from optimum.quanto.quantize import _quantize_submodule, set_module_by_name
from optimum.quanto.nn import QLinear, QConv2d
from optimum.quanto.tensor.activations import ActivationQBytesTensor, quantize_activation
from typing import Optional
from torch.nn.modules.module import  register_module_forward_hook, register_module_forward_pre_hook
from torch.overrides import TorchFunctionMode
from optimum.quanto import absmax_scale, QTensor
from optimum.quanto.calibrate import _updated_scale
import types
import re
import ipdb
from torchvision.ops import deform_conv2d



label2id = {
    'N/A': 83,
    'airplane': 5,
    'apple': 53,
    'backpack': 27,
    'banana': 52,
    'baseball bat': 39,
    'baseball glove': 40,
    'bear': 23,
    'bed': 65,
    'bench': 15,
    'bicycle': 2,
    'bird': 16,
    'boat': 9,
    'book': 84,
    'bottle': 44,
    'bowl': 51,
    'broccoli': 56,
    'bus': 6,
    'cake': 61,
    'car': 3,
    'carrot': 57,
    'cat': 17,
    'cell phone': 77,
    'chair': 62,
    'clock': 85,
    'couch': 63,
    'cow': 21,
    'cup': 47,
    'dining table': 67,
    'dog': 18,
    'donut': 60,
    'elephant': 22,
    'fire hydrant': 11,
    'fork': 48,
    'frisbee': 34,
    'giraffe': 25,
    'hair drier': 89,
    'handbag': 31,
    'horse': 19,
    'hot dog': 58,
    'keyboard': 76,
    'kite': 38,
    'knife': 49,
    'laptop': 73,
    'microwave': 78,
    'motorcycle': 4,
    'mouse': 74,
    'orange': 55,
    'oven': 79,
    'parking meter': 14,
    'person': 1,
    'pizza': 59,
    'potted plant': 64,
    'refrigerator': 82,
    'remote': 75,
    'sandwich': 54,
    'scissors': 87,
    'sheep': 20,
    'sink': 81,
    'skateboard': 41,
    'skis': 35,
    'snowboard': 36,
    'spoon': 50,
    'sports ball': 37,
    'stop sign': 13,
    'suitcase': 33,
    'surfboard': 42,
    'teddy bear': 88,
    'tennis racket': 43,
    'tie': 32,
    'toaster': 80,
    'toilet': 70,
    'toothbrush': 90,
    'traffic light': 10,
    'train': 7,
    'truck': 8,
    'tv': 72,
    'umbrella': 28,
    'vase': 86,
    'wine glass': 46,
    'zebra': 24
}


coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def _postprocessor(hm,wh,reg,metas, cat_spec_wh, K, scale, post0, post1, post2):
    dets = post0(hm, wh, reg=reg, cat_spec_wh=cat_spec_wh, K=K) #ctdet_decode
    results = []
    for det, meta in zip(dets, metas):
        # 배치당 
        scores = []; boxes =[]; labels=[];
        det = det.unsqueeze(0)
        det1 = post1(det, meta, scale) #Ctdet.post_process
        det2 = post2(det1)
        for cls in range(1,81): # coco 80 class 
            if det2[cls].shape[0]==0: continue
            box = det2[cls][:,:4] #xyxy 방식
            score = det2[cls][:,-1]
            label = label2id[coco_class_name[cls-1]]
            label = torch.tensor([label]*box.shape[0])
            scores.append(score)
            boxes.append(box)
            labels.append(label)
        boxes = torch.tensor(np.vstack(boxes))
        scores = torch.tensor(np.concat(scores))
        labels = torch.tensor(np.concat(labels))
        result = {'scores':scores, 'labels':labels, 'boxes':boxes}
        results.append(result)
    return results    


def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]

def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    if image_id is None:
        return {'image_id': '0', 'annotations': []}
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def train_transform(img, targets, processor):
    img = img.convert('RGB')    

    image_id = None
    categories = []
    areas= []
    bboxes = []

    for i,target in enumerate(targets):
        if i==0: image_id=str(target['image_id'])
        category = target['category_id']
        area = target['area']
        bbox = target['bbox']
        categories.append(category)
        areas.append(area)
        bboxes.append(bbox)
    annotations = format_image_annotations_as_coco(
        image_id, categories=categories, areas=areas, bboxes=bboxes
    )
    result = processor(images=img, annotations=annotations, return_tensors="pt")
    result.pop("pixel_mask", None)
    return result, targets


def eval_transform(img, targets, processor):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img, meta = processor(img)    
    return (img,meta), targets

def _collate_fn_eval(batch):       
    imgs    = []
    metas   = []
    targets = []
    for i,item in enumerate(batch):
        img  = item[0][0]
        meta = item[0][1]
        target = item[1]
        imgs.append(img.squeeze(0))
        metas.append(meta)
        target_=[]
        for t in target:
            del t['segmentation']
            del t['area']
            del t['iscrowd']
            target_.append(t)
        targets.append(target_)
    imgs = torch.stack(imgs)
    return imgs, targets, metas


def _collate_fn_train(batch):             
    images = []
    labels = []
    for i,item in enumerate(batch):       
        item   = item[0]
        image  = item['pixel_values'].squeeze(0)
        label = item['labels'][0]
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    return images, labels

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


def refacor_deformconv(model):
    for name, m in model.named_modules():
        if not isinstance(m,DeformConv): continue
        m2 =  DeformConv2(m)
        set_module_by_name(model,name,m2)
        m2.name = name
        for name, param in m.named_parameters():
            setattr(m, name, None)
            del param 
            
            
def is_match(name, patterns):
    for pattern in patterns:
        if pattern.startswith('re:'):
            regex = pattern[3:]
            if re.match(regex, name): 
                return True
        else:
            if pattern==name:
                return True
    return False      
            
def _quantize_deformconv(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude

    for name, m in model.named_modules():
        if include is not None:
            if is_match(name,include):
                if isinstance(m, deformconv2d):
                    qmodule = Qdeformconv2d.from_module(m, weights=weights, activations=activations, optimizer=optimizer)            
                    set_module_by_name(model, name, qmodule)
                    qmodule.name = name
                    for name, param in m.named_parameters():
                        setattr(m, name, None)
                        del param
            continue
        if is_match(name,exclude): continue


        # convtransepose2d 케이스 추가
        if isinstance(m, deformconv2d):
            qmodule = Qdeformconv2d.from_module(m, weights=weights, activations=activations, optimizer=optimizer)            
            set_module_by_name(model, name, qmodule)
            qmodule.name = name
            for name, param in m.named_parameters():
                setattr(m, name, None)
                del param


    # 전체적으로  output quantizer 제거
    # conv2d 는 input quantizer 붙임
    for name, m in model.named_modules():
        # if not isinstance(m,QModuleMixin): continue
        if not isinstance(m,Qdeformconv2d): continue
        m._quantize_hooks["input"].remove()
        m.disable_output_quantization()
        m._quantize_hooks.pop("output", None)
        m._quantize_hooks.pop("input", None)
        del m._buffers["output_scale"]     
        m._quantize_hooks["input"] = m.register_forward_pre_hook(quantize_input)
        m._save_to_state_dict = types.MethodType(_save_to_state_dict, m)


def quantize_input(module, input_):
    input, offset, mask = input_
    if isinstance(input, ActivationQBytesTensor):
        if input.qtype != module.activation_qtype:
            raise ValueError(
                "Models with heterogeneous quantized activations are not supported:"
                f" expectedmodule{M.activation_qtype.name} input but got {input.qtype.name} instead."
            )
    else:
        input = quantize_activation(input, qtype=module.activation_qtype, scale=module.input_scale)
    return input, offset, mask
        
def _save_to_state_dict(self, destination, prefix, keep_vars):
    if self.weight_qtype is None or not self.frozen:
        # Save standard weight Tensor
        destination[prefix + "weight"] = (
            self.weight if (self.weight is None or keep_vars) else self.weight.detach()
        )
    else:
        # Save QTensor using dedicated method
        self.weight.save_to_state_dict(destination, prefix + "weight.", keep_vars)
    if self.bias is not None:
        destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
    destination[prefix + "input_scale"] = self.input_scale if keep_vars else self.input_scale.detach()
    # destination[prefix + "output_scale"] = self.output_scale if keep_vars else self.output_scale.detach()
    


    
class Qdeformconv2d(QModuleMixin, deformconv2d):
    @classmethod
    def qcreate(
        cls,
        module,
        weights: qtype,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=module.bias is not None,
            dtype=module.weight.dtype,
            device=device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def forward(self, input, offset, mask):    
        # Use torchvision's deform_conv2d
        output = deform_conv2d(
            input    = input,
            offset   = offset,
            weight   = self.qweight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            mask     = mask
        )
        return output