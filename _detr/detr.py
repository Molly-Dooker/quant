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

from datasets import load_dataset

if __name__ == '__main__':
        # ipdb.set_trace()

        ds = load_dataset(path='rafaelpadilla/coco2017', cache_dir='/Data/Dataset/COCO', split='val')
        image = ds[0]['image']
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        device = 'cuda:1'
        # you can specify the revision tag if you don't want the timm dependency
        processor = DetrImageProcessor().from_pretrained("facebook/detr-resnet-50", revision="no_timm", size={"height": 800, "width": 800})
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        fold_frozen_bn_to_identity(model)

        model.to(device)
        model.eval()        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        ipdb.set_trace()
        # logit = outputs['logits']
        # box   = outputs['pred_boxes']
        # ipdb.set_trace()

        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                print(score,label)
        # 박스 좌표: post_process_object_detection은 픽셀 좌표를 반환합니다.
        # box 좌표는 [x1, y1, x2, y2] 형식입니다.
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="blue", width=2)
                text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
                # 텍스트는 박스의 왼쪽 위에 그립니다.
                draw.text((box[0], box[1]), text, fill="blue")

                # 결과 이미지를 파일로 저장합니다.
                draw_image.save("detection_result.jpg")
        ipdb.set_trace()

        # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        #         box = [round(i, 2) for i in box.tolist()]
        #         print(
        #                 f"Detected {model.config.id2label[label.item()]} with confidence "
        #                 f"{round(score.item(), 3)} at location {box}"
        #         )
                        
