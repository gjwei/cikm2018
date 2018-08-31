# coding: utf-8

# TODO:
"""
1. 将大写转化为小写形式
2. 将倒挂的问好和感叹号转为正挂的，并加上空格

2. 将两个训练样本中的西班牙语和英语分别提取出来。应为比赛只是使用西班牙语进行测试的，所以，我们先只用西班牙语进行训练

西班牙语中的倒感叹号放在句首用来表示语气的感叹或者疑问。用倒挂问号¿和倒挂感叹号¡的初衷，是方便读者在阅读一些较长的句子时，从一开始就能知道该句子是疑问句或感叹句。

"""
import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split


def to_lower(s):
    return s.lower()


def punctiation(s):
    s = re.sub('@', ' ', s)
    s = re.sub(':', ' ', s)
    s = re.sub('¿', '?', s)
    s = re.sub('¡', '!', s)
    pattern = re.compile(r'([(),.?!\'\"\-$&;|/*#{}@])')
    s = re.sub(r'([(),.?!\'\"\-$&;|/*#{}@])\1*', r'\1', s)
    s = pattern.sub(r' \1 ', s)
    # 下面是对文本的预处理，就放在这个函数中
    s = re.sub(r'([(),.?!\'\"\-$&;|/*#]){2,}', '\1', s)
    return s


def read_txt(path):
    with open(path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        return lines


def save_txt(path, name, data):
    with open(path + name + '.txt', 'wt', encoding='utf-8') as f:
        for line in data:
            f.write(line)
            

def concat_txt(path1, path2, is_train=True):
    """将预处理的数据保存到csv格式文件中"""
    f1 = open(path1, 'rt', encoding='utf-8')
    if is_train:
        f2 = open(path2, 'rt', encoding='utf-8')
    if is_train:
        all_data = f1.readlines() + f2.readlines()
    else:
        all_data = f1.readlines()
    
    data = []
    for line in all_data:
        line = line.strip().split('\t')
        text1 = re.sub(r'[0-9]+', ' número ', line[0])
        text2 = re.sub(r'[0-9]+', ' número ', line[1])
        label = line[2]
        
        data.append('\t'.join([label, text1, text2]) + '\n')

    f1.close()
    if is_train:
        f2.close()
        save_txt('../input/processing/', 'train', data)
    else:
        save_txt('../input/processing/', 'test', data)
    return data


if __name__ == '__main__':
    process_base_path = '../input/processing/'
    base_path = '../input/'
    if not os.path.exists(process_base_path):
        os.makedirs(process_base_path)
    
    train_en = base_path + 'cikm_english_train_20180516.txt'
    train_sp = base_path + 'cikm_spanish_train_20180516.txt'

    train_en_process = process_base_path + 'english_train_process.txt'
    train_sp_process = process_base_path + 'spanish_train_process.txt'

    # 将english_train中的英文和西班牙文划分开
    en_train = read_txt(train_en)
    ens = []
    sps = []
    for line in en_train:
        line = to_lower(line)
        line = punctiation(line)
        line = line.strip()
        line_arr = re.split('\t', line)
        ens.append('{}\t{}\t{}\n'.format(line_arr[0], line_arr[2], line_arr[4]))
        sps.append('{}\t{}\t{}\n'.format(line_arr[1], line_arr[3], line_arr[4]))
    save_txt(process_base_path, 'english_train_english', ens)
    save_txt(process_base_path, 'english_train_spanish', sps)

    del en_train

    sp_train = read_txt(train_sp)
    ens = []
    sps = []
    for line in sp_train:
        line = to_lower(line)
        line = punctiation(line)
        line = line.strip()
        line_arr = re.split('\t', line)
        sps.append('{}\t{}\t{}\n'.format(line_arr[0], line_arr[2], line_arr[4]))
        ens.append('{}\t{}\t{}\n'.format(line_arr[1], line_arr[3], line_arr[4]))

    save_txt(process_base_path, 'spanish_train_english', ens)
    save_txt(process_base_path, 'spanish_train_spanish', sps)
    
    print(u'对测试数据进行预处理，所有的label均设置为0')
    test_path = base_path + 'cikm_test_a_20180516.txt'
    test = read_txt(test_path)
    sps = []
    for line in test:
        line = to_lower(line)
        line = punctiation(line)
        line = line.strip()
        line_arr = re.split('\t', line)
        sps.append('{}\t{}\t{}\n'.format(line_arr[0], line_arr[1], 0))
    save_txt(process_base_path, 'spanish_test_spanish', sps)
    
    
    concat_txt(process_base_path + 'english_train_spanish.txt', process_base_path + 'spanish_train_spanish.txt', is_train=True)
    concat_txt(process_base_path + 'spanish_test_spanish.txt', path2=None, is_train=False)

    print('Done')




