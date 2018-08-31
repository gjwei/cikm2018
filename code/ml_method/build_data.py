#!/usr/bin/env python
#-*- coding:utf-8 -*- 
#Author: gjwei
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
# from build_data import read_data, save_data

names = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']


train_file = '../input/processing/spanish_train.txt'
test_file = '../input/processing/test.txt'

fasttext = '../fasttext/wiki.es.vec'

train = pd.read_csv(train_file, sep='\t', names=[names[-1], names[3], names[4]])
test = pd.read_csv(test_file, sep='\t', names=[names[-1], names[3], names[4]])

print(train.head())

all_quesions = set()


for questions in [train['question1'], train['question2'], test['question1'], test['question2']]:
    for q in questions:
        all_quesions.add(q)

all_quesions = list(all_quesions)

question_id = dict(zip(all_quesions, range(1, len(all_quesions) + 1)))

train['qid1'] = train['question1'].apply(lambda x: question_id[x])
train['qid2'] = train['question2'].apply(lambda x: question_id[x])

test['qid1'] = test['question1'].apply(lambda x: question_id[x])
test['qid2'] = test['question2'].apply(lambda x: question_id[x])



train['id'] = range(len(train))
test['id'] = range(len(test))

train = train[names]
test = test[names]

train.to_csv('./data/train.csv', sep='\t', index=False)
test.to_csv('./data/test.csv', sep='\t', index=False)

data = pd.concat([train, test], axis=0)
data.to_csv('./data/data.csv', sep='\t', index=False)


