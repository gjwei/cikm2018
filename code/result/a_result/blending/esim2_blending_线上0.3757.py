# coding: utf-8
# Author: gjwei
import glob
import numpy as np

def read_result(filename):
    result = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            result.append(float(line.strip()))

    return np.asarray(result, dtype=np.float32)

base_path = "./"
pattern = 'ESIM2_predict_result_dev_min_loss_0.25*'
files = glob.glob('{}{}'.format(base_path, pattern))

result = None

for i, file in enumerate(files):
    if i == 0:
        result = read_result(file)
    else:
        result += read_result(file)

result /= len(files)

with open("{}{}_blend.txt".format(base_path, pattern[:-1]), 'wt', encoding='utf-8')as f:
    for line in result:
        f.write(str(line) +'\n')
    print("Done")



