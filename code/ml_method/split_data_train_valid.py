# coding: utf-8
# Author: gjwei
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def split():
    base_path = "./data/"
    train_path = base_path + "train.csv"

    data_path = base_path + "quora_features.csv"

    train = pd.read_csv(train_path, sep='\t')
    data = pd.read_csv(data_path, sep=',')

    data = data.fillna(0.0)

    train_size = len(train)

    train = data.iloc[0: train_size, :]
    test = data.iloc[train_size:, :]

    train = shuffle(train)

    train_data, valid_data = train_test_split(train, test_size=0.15, shuffle=True)

    print(len(train.columns))

    save_path = "../input/processing/with_extra_features/"
    train_data.to_csv(save_path + 'train_data.csv', index=False, sep='\t')
    valid_data.to_csv(save_path + 'valid_data.csv', index=False, sep='\t')
    test.to_csv(save_path + 'test.csv', index=False, sep='\t')


if __name__ == '__main__':
    split()