import sys
from optimum.quanto import (
    qfloat8,
    qint4,
    qint8,
)
from transformers.models.detr.modeling_detr import DetrFrozenBatchNorm2d


import torch
import torch.nn as nn

def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]



def custom_transform(img, target, processor):
    src = processor(img.convert('RGB'), return_tensors="pt")    
    size = img.size[::-1]
    return (src['pixel_values'], size), target

def _collate_fn(batch):                
    pixel_values = []
    target       = []
    Size         = []
    for i,item in enumerate(batch):
        pixel_values.append(item[0][0].squeeze(0))
        Size.append(item[0][1])
        target_=[]
        for t in item[1]:
            del t['segmentation']
            del t['area']
            del t['iscrowd']
            target_.append(t)
        target.append(target_)
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target, Size



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

