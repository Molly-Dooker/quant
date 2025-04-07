import ipdb;
import json


ipdb.set_trace()
# json_file_path = "_stats/stat_default.json" 
json_file_path = "_stats/stat4_quantized.json" 
with open(json_file_path, "r") as f:
    data = json.load(f)


sorted_data = sorted(data.items(), key=lambda item: item[1]['std'], reverse=True)
for key, stats in sorted_data:
    print(f"{key:100}: std={stats['std']:.6f}, mean={stats['mean']:.6f}")