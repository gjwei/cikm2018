# coding: utf-8
# Author: gjwei
from sklearn.linear_model import LogisticRegression, Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
import numpy as np
from sklearn.metrics import log_loss



data = pd.read_csv("./data/quora_features.csv")

data = data.fillna(0.0)

data = data.drop(['wmd', 'norm_wmd'], axis=1)


train_size = len(pd.read_csv('./data/train.csv', sep='\t'))

X = data.iloc[0: train_size, 3:].astype(float).fillna(0.0)
Y = data.iloc[0: train_size, 2].astype(float).fillna(0.0)

# print((np.max(X)))
X = minmax_scale(X)

rf = RandomForestClassifier()
print(Y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# print(X_train.head())
# logreg.fit(X_train, y_train)
# print(cross_val_score(rf, X, Y))
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train,)

print(log_loss(y_test, rf.predict(X_test)))