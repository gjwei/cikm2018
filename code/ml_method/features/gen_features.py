# coding: utf-8
# Author: gjwei
import logging
import functools
from os.path import join as join_path
from collections import Counter, defaultdict
from itertools import count

import numpy as np
import pandas as pd

def stopwords():
    result = set()
    with open('../data/stopwords', 'wt', encoding='utf-8') as f:
        for line in f.readlines():
            result.add(line.strip())
    return set(result)


def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    r = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return r


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return len(wic) / len(uw)


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1'])*1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def tfidf_word_match_share(row, weights=None):

    q1words = {}
    for word in row['question1']:
        q1words[word] = 1

    q2words = {}
    for word in row['question2']:
        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = np.sum([weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words])

    total_weights = np.sum([weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words])

    if total_weights > 0:
        return shared_weights / total_weights
    else:
        return 0


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = np.sum([weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words])
    total_weights = np.sum([weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words])

    if total_weights > 0:
        return shared_weights / total_weights
    else:
        return 0


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

# counter
def compute_counters(train_df, test_df, **options):
    ques = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return (len(q_dict[row['question1']]))

    def q2_freq(row):
        return (len(q_dict[row['question2']]))

    def q1_q2_intersect(row):
        return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    def q1_q2_intersect_second_order(row):
        q1 = row['question1']
        q2 = row['question2']

        q1_neighbours = set(q_dict[q1])
        q1_neighbours_second_order = set(k for q in q1_neighbours for k in set(q_dict[q]) if k != q1 and k != q2)

        q2_neighbours = set(q_dict[q2])
        q2_neighbours_second_order = set(k for q in q2_neighbours for k in set(q_dict[q]) if k != q1 and k != q2)

        return len(q1_neighbours_second_order.intersection(q2_neighbours_second_order))

    train_df["intersect_q1_q2"] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
    train_df["intersect2_q1_q2"] = train_df.apply(q1_q2_intersect_second_order, axis=1, raw=True)
    train_df["freq_q1"] = train_df.apply(q1_freq, axis=1, raw=True)
    train_df["freq_q2"] = train_df.apply(q2_freq, axis=1, raw=True)

    test_df["intersect_q1_q2"] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
    test_df["intersect2_q1_q2"] = test_df.apply(q1_q2_intersect_second_order, axis=1, raw=True)
    test_df["freq_q1"] = test_df.apply(q1_freq, axis=1, raw=True)
    test_df["freq_q2"] = test_df.apply(q2_freq, axis=1, raw=True)

# distance
import distance
from sklearn.metrics import roc_auc_score

def levenshtein1(q1, q2):
    return distance.nlevenshtein(q1, q2, method=1)


def levenshtein2(q1, q2):
    return distance.nlevenshtein(q1, q2, method=2)


def sorencen(q1, q2):
    return distance.sorensen(q1, q2)


def compute_quality(train_df, feature):
    corr = train_df[["is_duplicate", feature]].corr().values.tolist()
    auc = roc_auc_score(train_df["is_duplicate"], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)

# Fuzzy
from fuzzywuzzy import fuzz


import FieldClass


def main():
    stops = stopwords()

    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    Fields = FieldClass()

    train_df = pd.read_csv("../data/train.csv", sep='\t')
    test_df = pd.read_csv("../data/test.csv", sep='\t')

    train_df['question1'] = train_df['question1'].map(lambda x: str(x).lower().split())
    train_df['question2'] = train_df['question2'].map(lambda x: str(x).lower().split())

    test_df['question1'] = test_df['question1'].map(lambda x: str(x).lower().split())
    test_df['question2'] = test_df['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    f = functools.partial(word_match_share, stops=stopwords())

    train_df['word_match'] = train_df.apply(f, axis=1, raw=True)
    test_df['word_match'] = test_df.apply(f, axis=1, raw=True)

    train_df['jaccard'] = train_df.apply(jaccard, axis=1, raw=True)
    test_df['jaccard'] = test_df.apply(jaccard, axis=1, raw=True)

    train_df['wc_diff'] = train_df.apply(wc_diff, axis=1, raw=True)
    test_df['wc_diff'] = test_df.apply(wc_diff, axis=1, raw=True)

    train_df['wc_diff_unique'] = train_df.apply(wc_diff_unique, axis=1, raw=True)
    test_df['wc_diff_unique'] = test_df.apply(wc_diff_unique, axis=1, raw=True)

    train_df['wc_ratio_unique'] = train_df.apply(wc_ratio_unique, axis=1, raw=True)
    test_df['wc_ratio_unique'] = test_df.apply(wc_ratio_unique, axis=1, raw=True)

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    train_df['wc_diff_unq_stop'] = train_df.apply(f, axis=1, raw=True)
    test_df['wc_diff_unq_stop'] = test_df.apply(f, axis=1, raw=True)

    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    train_df["wc_ratio_unique_stop"] = train_df.apply(f, axis=1, raw=True)
    test_df["wc_ratio_unique_stop"] = test_df.apply(f, axis=1, raw=True)

    train_df["same_start"] = train_df.apply(same_start_word, axis=1, raw=True)
    test_df["same_start"] = test_df.apply(same_start_word, axis=1, raw=True)

    train_df["char_diff"] = train_df.apply(char_diff, axis=1, raw=True)
    test_df["char_diff"] = test_df.apply(char_diff, axis=1, raw=True)

    f = functools.partial(char_diff_unique_stop, stops=stops)
    train_df["char_diff_unq_stop"] = train_df.apply(f, axis=1, raw=True)
    test_df["char_diff_unq_stop"] = test_df.apply(f, axis=1, raw=True)

    train_df["total_unique_words"] = train_df.apply(total_unique_words, axis=1, raw=True)
    test_df["total_unique_words"] = test_df.apply(total_unique_words, axis=1, raw=True)

    f = functools.partial(total_unq_words_stop, stops=stops)
    train_df["total_unq_words_stop"] = train_df.apply(f, axis=1, raw=True)
    test_df["total_unq_words_stop"] = test_df.apply(f, axis=1, raw=True)

    train_df["char_ratio"] = train_df.apply(char_ratio, axis=1, raw=True)
    test_df["char_ratio"] = test_df.apply(char_ratio, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share, weights=weights)
    train_df["tfidf_wm"] = train_df.apply(f, axis=1, raw=True)
    test_df["tfidf_wm"] = test_df.apply(f, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    train_df["tfidf_wm_stops"] = train_df.apply(f, axis=1, raw=True)
    test_df["tfidf_wm_stops"] = test_df.apply(f, axis=1, raw=True)
    
    # counter
    compute_counters(train_df, test_df)

    # distance

    train_df["levenstein1"] = train_df.apply(lambda r: levenshtein1(r[Fields.question1], r[Fields.question2]),
                                                  axis=1)
    test_df[Fields.levenstein1] = test_df.apply(lambda r: levenshtein1(r[Fields.question1], r[Fields.question2]),
                                                axis=1)
    train_df[Fields.levenstein2] = train_df.apply(lambda r: levenshtein2(r[Fields.question1], r[Fields.question2]),
                                                  axis=1)
    test_df[Fields.levenstein2] = test_df.apply(lambda r: levenshtein2(r[Fields.question1], r[Fields.question2]), axis=1)
    train_df[Fields.sorensen] = train_df.apply(lambda r: sorencen(r[Fields.question1], r[Fields.question2]), axis=1)
    test_df[Fields.sorensen] = test_df.apply(lambda r: sorencen(r[Fields.question1], r[Fields.question2]), axis=1)


    # fuzzy
    train_df[Fields.qratio] = train_df.apply(
        lambda row: fuzz.QRatio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    test_df[Fields.qratio] = test_df.apply(
        lambda row: fuzz.QRatio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    quality_qratio = compute_quality(train_df, Fields.qratio)

    train_df[Fields.wratio] = train_df.apply(
        lambda row: fuzz.WRatio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    test_df[Fields.wratio] = test_df.apply(
        lambda row: fuzz.WRatio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    quality_wratio = compute_quality(train_df, Fields.wratio)

    train_df[Fields.partial_ratio] = train_df.apply(
        lambda row: fuzz.partial_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    test_df[Fields.partial_ratio] = test_df.apply(
        lambda row: fuzz.partial_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    quality_partial_ratio = compute_quality(train_df, Fields.partial_ratio)

    train_df[Fields.partial_token_set_ratio] = train_df.apply(
        lambda row: fuzz.partial_token_set_ratio(str(row[Fields.question1]), str(row[Fields.question2])),
        axis=1)
    test_df[Fields.partial_token_set_ratio] = test_df.apply(
        lambda row: fuzz.partial_token_set_ratio(str(row[Fields.question1]), str(row[Fields.question2])),
        axis=1)
    quality_partial_token_set_ratio = compute_quality(train_df, Fields.partial_token_set_ratio)

    train_df[Fields.partial_token_sort_ratio] = train_df.apply(
        lambda row: fuzz.partial_token_sort_ratio(str(row[Fields.question1]), str(row[Fields.question2])),
        axis=1)
    test_df[Fields.partial_token_sort_ratio] = test_df.apply(
        lambda row: fuzz.partial_token_sort_ratio(str(row[Fields.question1]), str(row[Fields.question2])),
        axis=1)
    quality_partial_token_sort_ratio = compute_quality(train_df, Fields.partial_token_sort_ratio)

    train_df[Fields.token_set_ratio] = train_df.apply(
        lambda row: fuzz.token_set_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    test_df[Fields.token_set_ratio] = test_df.apply(
        lambda row: fuzz.token_set_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    quality_token_set_ratio = compute_quality(train_df, Fields.token_set_ratio)

    train_df[Fields.token_sort_ratio] = train_df.apply(
        lambda row: fuzz.token_sort_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    test_df[Fields.token_sort_ratio] = test_df.apply(
        lambda row: fuzz.token_sort_ratio(str(row[Fields.question1]), str(row[Fields.question2])), axis=1)
    quality_token_sort_ratio = compute_quality(train_df, Fields.token_sort_ratio)

    train_df.to_csv(train_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)


    