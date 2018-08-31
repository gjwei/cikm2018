# coding: utf-8
# Author: gjwei
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss


data_path = "../../input/processing/dumps/"

result_path = "./result/"

train_data = pd.read_csv(data_path + 'train.csv', sep='\t')
test_data = pd.read_csv(data_path + 'test.csv', sep='\t')

drop_columns = 'id	qid1	qid2	question1	question2	is_duplicate'.split('\t')
print(drop_columns)

y_train = train_data['is_duplicate'].values

X_train = train_data.drop(drop_columns, axis=1)
X_test = test_data.drop(drop_columns, axis=1)

X_test = X_test[X_train.columns]

print(set(X_test.columns) - set(X_train.columns))
assert (X_train.columns == X_test.columns).all()

# 5 fold code:
n_fold = 5

bst = None

losses = []

for i in range(n_fold):
    print('fold {} / {}'.format(i + 1, n_fold))
    shuffle(train_data)
    test_size = 1.0 / n_fold

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)

    # # UPDownSampling
    # pos_train = X_train[y_train == 1]
    # neg_train = X_train[y_train == 0]
    # X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
    # y_train = np.array(
    #     [0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    # print(np.mean(y_train))
    # del pos_train, neg_train

    # pos_valid = X_valid[y_valid == 1]
    # neg_valid = X_valid[y_valid == 0]
    # X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    # y_valid = np.array(
    #     [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    # print(np.mean(y_valid))
    # del pos_valid, neg_valid

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    # params['nthread '] = 8
    params['max_depth'] = 7
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    params['scale_pos_weight'] = 0.2

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)
    d_test = xgb.DMatrix(X_test)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=2)
    valid_loss = log_loss(y_valid, bst.predict(d_valid))
    print(valid_loss)
    losses.append(valid_loss)


    if i == 0:
        test_preds = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)
    else:
        test_preds += bst.predict(d_test, ntree_limit=bst.best_ntree_limit)

test_preds /= n_fold

mean_loss = np.mean(losses)
print('mean valid loss is {}'.format(mean_loss))
# save_test_preds
save_name = "./result/xgboost_{}_fold_{:.4f}valid_loss.txt".format(n_fold, mean_loss)

with open(save_name, 'wt', encoding='utf-8') as f:
    for line in test_preds:
        f.write("{}\n".format(line))


