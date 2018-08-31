# coding: utf-8
# Author: gjwei

import numpy as np

from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis

import gensim
import logging
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='progress')

from Fields import FieldClass

fasttext = '../../fasttext/wiki.es.vec'

stop_words = set()
with open('../data/stopwords.txt', 'rt', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.add(line.strip())

class Glove(object):
    def __init__(self, model):
        self.model = model

    def features(self, q1, q2):
        q1 = str(q1).lower().split()
        q2 = str(q2).lower().split()
        q1 = [w for w in q1 if w not in stopwords]
        q2 = [w for w in q2 if w not in stopwords]

        wmd = min(self.model.wmdistance(q1, q2), 10)

        q1vec = self.sent2vec(q1)
        q2vec = self.sent2vec(q2)

        if q1vec is not None and q2vec is not None:
            cos = cosine(q1vec, q2vec)
            city = cityblock(q1vec, q2vec)
            jacc = jaccard(q1vec, q2vec)
            canb = canberra(q1vec, q2vec)
            eucl = euclidean(q1vec, q2vec)
            mink = minkowski(q1vec, q2vec, 3)
            bray = braycurtis(q1vec, q2vec)

            q1_skew = skew(q1vec)
            q2_skew = skew(q2vec)
            q1_kurt = kurtosis(q1vec)
            q2_kurt = kurtosis(q2vec)

        else:
            cos = -1
            city = -1
            jacc = -1
            canb = -1
            eucl = -1
            mink = -1
            bray = -1

            q1_skew = 0
            q2_skew = 0
            q1_kurt = 0
            q2_kurt = 0

        return wmd, cos, city, jacc, canb, eucl, mink, bray, q1_skew, q2_skew, q1_kurt, q2_kurt

    def sent2vec(self, words):
        M = []
        for w in words:
            try:
                M.append(self.model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        norm = np.sqrt((v ** 2).sum())
        if norm > 0:
            return v / np.sqrt((v ** 2).sum())
        else:
            return None

if __name__ == '__main__':
    Fields = FieldClass()

    glove = gensim.models.KeyedVectors.load_word2vec_format(fasttext, binary=False)
    glove.init_sims(replace=True)
    processor = Glove(glove)

    logging.warning('Computing train features')

    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    train_df = pd.read_csv("../data/train.csv", sep='\t')
    test_df = pd.read_csv("../data/test.csv", sep='\t')

    train_df[Fields.glove_wmd], \
    train_df[Fields.glove_cos], \
    train_df[Fields.glove_city], \
    train_df[Fields.glove_jacc], \
    train_df[Fields.glove_canb], \
    train_df[Fields.glove_eucl], \
    train_df[Fields.glove_mink], \
    train_df[Fields.glove_bray], \
    train_df[Fields.glove_skew_q1], \
    train_df[Fields.glove_skew_q2], \
    train_df[Fields.glove_kurt_q1], \
    train_df[Fields.glove_kurt_q2] = \
        zip(*train_df.progress_apply(lambda r: processor.features(r['question1'], r['question2']), axis=1))

    logging.warning('Computing test features')
    test_df[Fields.glove_wmd], \
    test_df[Fields.glove_cos], \
    test_df[Fields.glove_city], \
    test_df[Fields.glove_jacc], \
    test_df[Fields.glove_canb], \
    test_df[Fields.glove_eucl], \
    test_df[Fields.glove_mink], \
    test_df[Fields.glove_bray], \
    test_df[Fields.glove_skew_q1], \
    test_df[Fields.glove_skew_q2], \
    test_df[Fields.glove_kurt_q1], \
    test_df[Fields.glove_kurt_q2] = \
        zip(*test_df.progress_apply(lambda r: processor.features(r['question1'], r['question2']), axis=1))

    train_df.to_csv(train_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)