import ipdb;
import json
import sys
import numpy as np
eps = sys.float_info.epsilon



_rmse = '_stats/RMSE.json'
with open(_rmse, "r") as f:
    rmse = json.load(f)
for key in rmse.keys():
    rmse[key] = np.mean(rmse[key]).item()


_quantized_outlier = "_stats/quantized_outlier.json" 
_quantized = "_stats/quantized.json" 
_default_outlier   = "_stats/default_outlier.json" 
with open(_quantized_outlier, "r") as f:
    quantized_outlier = json.load(f)
with open(_quantized, "r") as f:
    quantized = json.load(f)
with open(_default_outlier, "r") as f:
    default_outlier = json.load(f)

for key in quantized.keys():
    quantized[key]['quantized_outlier_ratio']=quantized_outlier[key]['outlier_ratio'] 
    quantized[key]['default_outlier_ratio']=default_outlier[key]['outlier_ratio'] 
    quantized[key]['rmse']=rmse[key]
    


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
