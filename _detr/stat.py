import ipdb;
import json
import sys

eps = sys.float_info.epsilon

json_file_path = "_stats/stat_default.json" 
json_file_path2 = "_stats/stat4_quantized.json" 
with open(json_file_path, "r") as f:
    default = json.load(f)

with open(json_file_path2, "r") as f:
    quant = json.load(f)
ipdb.set_trace()

for key in default.keys():
    std_default = default[key]['std'] 
    std_quant   = quant[key]['std']
    ratio       = std_quant/(std_default+eps)
    quant[key]['ratio'] =  ratio


sorted_data = sorted(quant.items(), key=lambda item: item[1]['ratio'], reverse=True)
for key, stats in sorted_data:
    print(f"{key:100}: std={stats['std']:.6f}, ratio={stats['ratio']:.6f}")