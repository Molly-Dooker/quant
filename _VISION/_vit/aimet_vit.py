import argparse
import sys
import json
import itertools
import math
import ipdb
from loguru import logger
from tqdm import tqdm
from _util import ipdb_sys_excepthook, keyword_to_itype, _transform

import torch
import evaluate
from accelerate import init_empty_weights
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
)
from optimum.quanto import (
    QTensor,
    freeze,
    qfloat8,
    qint4,
    qint8,
    quantization_map
)
from _quanto import _quantize, _requantize, _Calibration

def logger_enable(prefix=''):
    def console_filter(record):
        # extra에 file_only가 True인 경우 콘솔 출력 제외
        return not record["extra"].get("file_only", False)
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add(sys.stdout, level="INFO", format=LOG_FORMAT, filter=console_filter)
    logger.add("_logs/log", rotation="500 MB", level="INFO", format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)




def eval(model, device, test_loader, prefix=''):
    model.to(device)
    model.eval()
    metric = evaluate.load("accuracy")
    CORRECT = 0
    TOTAL   = 0
    with torch.no_grad():
        for batch in tqdm(test_loader,desc='eval...'):
            data, target = batch["pixel_values"], batch["labels"]
            data= data.to(device)
            output = model(data)
            output = output.argmax(-1).cpu()
            metric.add_batch(predictions=output,references=target)
            correct = (target==output).sum().item()
            CORRECT+= correct
            TOTAL  += target.shape[0]
            
            print(f'{CORRECT}/{TOTAL} : {CORRECT/TOTAL:.3f}')
            

    acc = metric.compute()['accuracy']
    logger.info(f'{prefix} model acc : {acc*100:.2f}%')

def calibrate(model, device, dataloader, num=10000):
    model.to(device)
    model.eval()
    iter =  min(math.ceil(num/dataloader.batch_size), dataloader.__len__())
    with torch.no_grad():
        for batch in tqdm(itertools.islice(dataloader, iter), total=iter, desc="calibrating..."):
            # ipdb.set_trace()
            data = batch["pixel_values"].to(device)
            _ = model(data)

import torchvision
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from BOS_util import to_qdq_torch,save_graph


from torch import nn
import torch.nn.functional as F

from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping, get_default_qconfig, get_default_qat_qconfig, QConfigMapping, QConfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.nn import MultiheadAttention


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = dropout  # attention 확률 드롭아웃 값

        # Q, K, V를 위한 개별 Linear 레이어 (bias 사용 여부 선택 가능)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 출력 투사 Linear 레이어
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                need_weights=True, average_attn_weights=True, is_causal=False):
        B, L, _ = query.shape  # batch, seq_len, embed_dim (batch_first=True 가정)
        # 1. Q, K, V 각각 Linear 변환
        Q = self.q_proj(query)   # shape: [B, L, embed_dim]
        K = self.k_proj(key)     # shape: [B, L, embed_dim]
        V = self.v_proj(value)   # shape: [B, L, embed_dim]

        # 2. num_heads로 분할하여 shape을 [B, num_heads, L, head_dim]으로 변환
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]

        # 3. Scaled Dot-Product Attention 수행 (PyTorch F.scaled_dot_product_attention 활용)
        # 드롭아웃은 학습 시에만 attn_weight에 적용되도록 dropout_p 설정
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, 
                                                     dropout_p=dropout_p, is_causal=is_causal)
        # attn_output shape: [B, num_heads, L, head_dim]

        # 4. 각 head별 출력 결합 및 최종 선형 출력 프로젝션
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)  # [B, L, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, L, embed_dim]

        # 5. 필요 시 어텐션 가중치 계산 (need_weights=True인 경우)
        if need_weights:
            # Q와 K를 사용하여 어텐션 가중치 행렬 계산 (소프트맥스 적용 전 마스킹 처리)
            # (주의: Q, K는 현재 [B, num_heads, L, head_dim])
            # QK^T / sqrt(d) 계산
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, L, L]
            # 주어진 마스크 처리
            if attn_mask is not None:
                # attn_mask가 bool 마스크인 경우 False인 위치에 -inf 추가
                # (bool에서 True는 attention 차단 위치를 의미)
                if attn_mask.dtype == torch.bool:
                    # attn_mask shape: [L, L] 또는 [B * num_heads, L, L]
                    # batch별 동일 마스크 가정 시 2D 가능. 필요 시 broadcast 조정.
                    attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
                else:
                    # attn_mask가 이미 -inf 또는 가중치로 주어진 경우 직접 더함
                    attn_scores = attn_scores + attn_mask
            if key_padding_mask is not None:
                # key_padding_mask: [B, L]에서 패딩(True)인 위치에 -inf 부여
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.view(B, 1, 1, L), float('-inf')
                )
            # Softmax로 확률로 변환 (dim=-1)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, L, L]
            if average_attn_weights:
                # 모든 헤드에 대해 평균 내기 -> [B, L, L]
                attn_weights = attn_weights.mean(dim=1)
            # (average_attn_weights=False이면 [B, num_heads, L, L] 그대로 반환)
            return attn_output, attn_weights
        else:
            return attn_output, None


def replace_attention_modules(model):
    for name, module in list(model._modules.items()):  # 하위 모듈 순회
        if isinstance(module, nn.MultiheadAttention):
            # 기존 모듈 속성 가져오기
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            dropout = module.dropout

            # CustomMultiheadAttention으로 교체할 모듈 생성
            new_module = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=(module.in_proj_bias is not None))
            new_module = new_module.to(module.out_proj.weight.device)  # 기존 모듈과 동일한 장치로 이동

            # 가중치 복사: in_proj_weight를 3등분하여 q, k, v에 복사
            if module.in_proj_weight is not None:
                # 모듈의 in_proj_weight, in_proj_bias를 텐서 쪼개기
                W = module.in_proj_weight.data  # shape [3*D, D]
                b = module.in_proj_bias.data if module.in_proj_bias is not None else None
                D = embed_dim
                # q_proj 가중치 및 바이어스
                new_module.q_proj.weight.data.copy_(W[0:D, :])
                new_module.k_proj.weight.data.copy_(W[D:2*D, :])
                new_module.v_proj.weight.data.copy_(W[2*D:3*D, :])
                if b is not None:
                    new_module.q_proj.bias.data.copy_(b[0:D])
                    new_module.k_proj.bias.data.copy_(b[D:2*D])
                    new_module.v_proj.bias.data.copy_(b[2*D:3*D])
            else:
                # 만약 module이 별도 q_proj_weight 등을 갖고 있다면 (embed_dim 다를 때)
                new_module.q_proj.weight.data.copy_(module.q_proj_weight)
                new_module.k_proj.weight.data.copy_(module.k_proj_weight)
                new_module.v_proj.weight.data.copy_(module.v_proj_weight)
                if module.in_proj_bias is not None:
                    new_module.q_proj.bias.data.copy_(module.in_proj_bias[0:D])
                    new_module.k_proj.bias.data.copy_(module.in_proj_bias[D:2*D])
                    new_module.v_proj.bias.data.copy_(module.in_proj_bias[2*D:3*D])

            # out_proj 가중치와 바이어스 복사
            new_module.out_proj.weight.data.copy_(module.out_proj.weight.data)
            if module.out_proj.bias is not None:
                new_module.out_proj.bias.data.copy_(module.out_proj.bias.data)

            # 모델에 모듈 교체
            model._modules[name] = new_module

        else:
            # 재귀적으로 내부 모듈 치환
            replace_attention_modules(module)

def main(args):
    logger_enable(args.prefix) 
    ds = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir, split=args.split)
    prepared_ds = ds.with_transform(lambda batch: _transform(batch, torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()))    
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # eval(model, args.device, dataloader,'default') #81.88    

    
    
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).eval()
    replace_attention_modules(model)
    # eval(model, args.device, dataloader,'default') #81.88    

    dummy_input = torch.randn(1, 3, 224, 224)    
    qconfig = get_default_qconfig()

    qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear,qconfig).set_object_type(torch.nn.Conv2d,qconfig).set_object_type(torch.nn.ReLU,qconfig).set_object_type(torch.nn.LayerNorm,None)\
        .set_module_name('encoder.layers.encoder_layer_5.self_attention.out_proj',None)\
        .set_module_name('encoder.layers.encoder_layer_9.self_attention.q_proj',None)\
        .set_module_name('heads.head',None)

    prepared_model = prepare_fx(model, qconfig_mapping, dummy_input) 
    calibrate(prepared_model,args.device,dataloader,4000)
    prepared_model.to('cpu')
    q_model = convert_fx(prepared_model)
    q_model.eval()    
    jit_model = torch.jit.trace(q_model,dummy_input)
    
    # jit_model.save('vit_jit.pt')
    # jit_model = torch.jit.load('vit_jit.pt')
    eval(jit_model, 'cpu', dataloader,'quantized')

    

    # model = prepare_model(model)
    # fold_all_batch_norms(model, dummy_input.shape, dummy_input=dummy_input)
    # model.to(args.device); dummy_input = dummy_input.to(args.device)



    # exclude = ['vit.encoder.layer.5.output.dense',
    #             'vit.encoder.layer.9.attention.attention.query']
    # if args.exclude is not None:
    #     exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
    #     if args.exclude=='': exclude = []
    # logger.info(f'exclude : {exclude}')   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ViT")
    parser.add_argument("--prefix", type=str, default="ViT")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", type=str, default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--cache_dir", type=str, default='/Data/Dataset/ImageNet')
    parser.add_argument("--saveroot", type=str, default='./_model')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=4, help="The device to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='Disable eval mode')

    parser.add_argument('--default', action='store_true', help='Enable stat mode')
    parser.add_argument('--no-default', dest='default', action='store_false')

    parser.add_argument('--exclude', type=str)

    args = parser.parse_args()
    args.device = f'cuda:{args.device}'   

    main(args)