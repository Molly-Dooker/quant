import ipdb;
import json
import sys
import numpy as np
import pandas as pd 
eps = sys.float_info.epsilon


_total = '_stats/total.json'
with open(_total, 'r', encoding='utf-8') as f:
    total = json.load(f)
ipdb.set_trace()

_mAP = '_stats/mAP.json'
with open(_mAP, 'r', encoding='utf-8') as f:
    mAP = json.load(f)

for key in total.keys():
    mAP50   = mAP[key]['mAP50']
    mAP5095 = mAP[key]['mAP50-95']
    total[key]['mAP50']= float(mAP50)
    total[key]['mAP5095']= float(mAP5095)

with open('_stats/total.json', 'w', encoding='utf-8') as json_file:
    json.dump(total, json_file, ensure_ascii=False, indent=4)

# JSON 데이터를 DataFrame으로 변환 (key는 index로, value는 딕셔너리)
df = pd.DataFrame.from_dict(total, orient='index')

# index를 컬럼으로 변환하고, 컬럼명을 'key'로 지정
df.reset_index(inplace=True)
df.rename(columns={'index': 'key'}, inplace=True)

# DataFrame 확인 (선택사항)
print(df.head())

# CSV 파일로 저장 (index 제외)
df.to_csv('_stats/total.csv', encoding='utf-8')

_rmse = '_stats/RMSE.json'
with open(_rmse, "r") as f:
    rmse = json.load(f)
_quantized_outlier = "_stats/quantized_outlier.json" 
with open(_quantized_outlier, "r") as f:
    quantized_outlier = json.load(f)
_quantized = "_stats/quantized.json"
with open(_quantized, "r") as f:
    quantized = json.load(f)


for key in rmse.keys():
    rmse_    = rmse[key]
    outlier_ = quantized_outlier[key]['outlier_ratio'] 
    std_     = quantized[key]['std']
    rmse[key]= {
        'rmse':rmse_,
        'outlier':outlier_,
        'std':std_
    }
with open('_stats/total.json', 'w', encoding='utf-8') as json_file:
    json.dump(rmse, json_file, ensure_ascii=False, indent=4)

    


sorted_data = sorted(quantized.items(), key=lambda item: item[1]['rmse'], reverse=True)
ipdb.set_trace()
for key, stats in sorted_data:
    print(f"{key:100}: rmse={stats['rmse']:.6f},    outlier_ratio={stats['quantized_outlier_ratio']:.6f}")

# json_file_path = "_stats/stat_default.json" 
# json_file_path2 = "_stats/stat4_quantized.json" 
# with open(json_file_path, "r") as f:
#     default = json.load(f)

# with open(json_file_path2, "r") as f:
#     quant = json.load(f)
# ipdb.set_trace()

# for key in default.keys():
#     std_default = default[key]['std'] 
#     std_quant   = quant[key]['std']
#     ratio       = std_quant/(std_default+eps)
#     quant[key]['ratio'] =  ratio


# sorted_data = sorted(quant.items(), key=lambda item: item[1]['ratio'], reverse=True)
# for key, stats in sorted_data:
#     print(f"{key:100}: std={stats['std']:.6f}, ratio={stats['ratio']:.6f}")


# if STAT:
#     logger.info('STAT start')
#     with open('_stats/stat4_quantized.json', "r") as f:
#         data = json.load(f)
#     from _stats import register_hook_default, unregister_hook_default, get_stats, get_outlier, register_hook_quantized, unregister_hook_quantized
#     register_hook_quantized(model,data)
#     calibrate(model, args.device, dataloader)  
#     # ipdb.set_trace()  
#     outs = get_outlier()
#     os.makedirs('_stats',exist_ok=True)
#     with open(f'_stats/{args.prefix}_quantized_outlier.json', "w") as f:
#         json.dump(outs, f, indent=2)
#     unregister_hook_quantized(model)
