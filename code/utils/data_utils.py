# coding: utf-8
# Author: gjwei
import numpy as np


def load_vocab(path):
    vocabs = []
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            vocabs.append(line.strip())
    return vocabs

def load_char_vocab(path):
    vocabs = ['@']
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            vocabs.append(line[0])
    return vocabs

def onehot(labels, num_classes):
    size = len(labels)
    labels = labels.astype(int)
    result = np.zeros(shape=(size, num_classes), dtype=np.float32)
    for i in range(size):
        result[i, labels[i]] = 1
    return result


def read_data(path):
    data = []
    with open(path, 'rt', encoding='utf-8') as f:
        data = f.readlines()
    return data


def save_data(path, data, newline=False):
    if newline:
        single = '\n'
    else:
        single = ''
    with open(path, 'wt', encoding='utf-8') as f:
        for line in data:
            f.write(line + single)
    print('save {} done'.format(path))


def float_extra_feature(extras):
    result = []
    for feature in extras:
        if len(feature) == 0:
            result.append(0.0)
        else:
            result.append(float(feature))

    return result

if __name__ == '__main__':
    labels = np.array([1, 0, 1, 0])
    print(onehot(labels, 2))

