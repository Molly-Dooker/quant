from collections import defaultdict
import numpy as np
import torch 
from torch.nn import Conv2d, Linear
from optimum.quanto.nn import QLinear, QConv2d
class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_batch(self, x: torch.Tensor):
        """
        x: 배치 단위의 activation 텐서. (예: [batch, ...])
        전체 요소(flattened) 기준으로 평균과 분산을 계산하고, 누적 통계를 업데이트합니다.
        """
        # 배치를 1차원으로 flatten
        # x_flat = x.detach().view(-1)
        n_batch =  x.numel()
        if n_batch == 0:
            return
        
        # 배치의 평균과 모집단 분산 (분산 계산시 unbiased=False)
        batch_mean = x.mean().item()
        # 모집단 분산: M2_batch = variance * n_batch
        batch_var = x.var(unbiased=False).item()
        batch_M2 = batch_var * n_batch

        
        if self.n == 0:
            # 첫 배치일 경우, 누적 통계를 초기화
            self.n = n_batch
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            # 기존 누적 통계와 새로운 배치 통계를 결합하는 공식
            N = self.n
            n_total = N + n_batch
            delta = batch_mean - self.mean
            # 새로운 평균: 가중 평균
            new_mean = (N * self.mean + n_batch * batch_mean) / n_total
            # 새로운 M2: 기존의 M2와 배치 M2를 합하고, 두 평균의 차이에 대한 보정항을 더함
            new_M2 = self.M2 + batch_M2 + (delta ** 2) * N * n_batch / n_total
            
            self.n = n_total
            self.mean = new_mean
            self.M2 = new_M2

    def finalize(self):
        """
        최종적으로 누적된 통계에서 평균과 sample variance, 표준편차를 계산합니다.
        sample variance: M2 / (n - 1) (n > 1)
        """
        if self.n < 2:
            return self.mean, float('nan')
        sample_var = self.M2 / (self.n - 1)
        std = np.sqrt(sample_var).item()
        return self.mean, std

module_stats = defaultdict(OnlineStats)

def online_stats_hook(module, inputs):
    activation = inputs[0]
    module_stats[id(module)].update_batch(activation)


def register_hook_default(model):
    for name, m in model.named_modules():
        if not isinstance(m, (Conv2d, Linear)): continue
        module_stats[id(m)].name = name
        m._stat_hooks = {}
        m._stat_hooks["input"] = m.register_forward_pre_hook(online_stats_hook)
        
def unregister_hook_default(model):
    for name, m in model.named_modules():
        if not isinstance(m, (Conv2d, Linear)): continue
        m._stat_hooks["input"].remove()
        del m._stat_hooks

def register_hook_quantized(model):
    for name, m in model.named_modules():
        if not isinstance(m, (QConv2d, QLinear)): continue
        module_stats[id(m)].name = name
        m._stat_hooks = {}
        m._stat_hooks["input"] = m.register_forward_pre_hook(online_stats_hook)
        
def unregister_hook_quantized(model):
    for name, m in model.named_modules():
        if not isinstance(m, (QConv2d, QLinear)): continue
        m._stat_hooks["input"].remove()
        del m._stat_hooks

def get_stats():
    stats_by_module_name = {}
    for module_id, stats in module_stats.items():
        mean, std = stats.finalize()
        stats_by_module_name[stats.name]={
            "mean": mean,
            "std": std
        }
    return stats_by_module_name



## how to use

# if STAT:
#     from _stats import register_hook_default, unregister_hook_default, get_stats
#     register_hook_default(model)
#     calibrate(model, args.device, dataloader)
#     stats = get_stats()
#     os.makedirs('_stats',exist_ok=True)
#     with open(f'_stats/{args.prefix}_default.json', "w") as f:
#         json.dump(stats, f, indent=2)
#     unregister_hook_default(model)


# if STAT:
#     from _stats import register_hook_quantized, unregister_hook_quantized, get_stats
#     register_hook_quantized(model)    
#     eval(model, args.device, dataloader, processor, 'quantized')    
#     stats = get_stats()        
#     os.makedirs('_stats',exist_ok=True)
#     with open(f'_stats/{args.prefix}_quantized.json', "w") as f:
#         json.dump(stats, f, indent=2)
#     unregister_hook_quantized(model)  
# else:
#     eval(model, args.device, dataloader, processor, 'quantized')