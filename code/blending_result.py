# coding: utf-8
# Author: gjwei
import glob

import numpy as np
import os


def read_result(path):
    result = []
    with open(path, 'rt', encoding='utf-8') as f:
        data = f.readlines()
        assert len(data) == 10000

        for line in data:
            result.append(float(line.strip()))
    
    return np.asarray(result, dtype=np.float32)


def save_blending_result(path, result):
    with open(path, 'wt', encoding='utf-8') as f:
        for line in result:
            f.write("{}\n".format(line))
    print("Save file {} Done".format(path))


path = glob.glob('./result/_MPCNN*') + glob.glob("./result/_ESIM*")
for p in path:
    print(p)
 
save_path = "../submits/b/"

# save_name = '_'.join([file[7:-4].split("_")[0] for file in path]) + '.txt'
if not os.path.exists(save_path):
    os.makedirs(save_path)


save_name = "models_{}_blend_b.txt".format(len(path))
result = None
for i in range(len(path)):
    if i == 0:
        result = read_result(path[i])
    else:
        result += read_result(path[i])

result /= len(path)

save_blending_result("{}{}".format(save_path, save_name), result)
