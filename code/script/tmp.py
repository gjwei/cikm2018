# coding: utf-8
# Author: gjwei
import pandas as pd

train = pd.read_csv('../input/processing/train.csv', sep='\t')
print(len(train))