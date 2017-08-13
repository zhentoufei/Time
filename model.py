# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 20:49'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'model.py' 


import pandas as pd
import numpy as np
import xgboost as xgb


from sklearn.ensemble import GradientBoostingRegressor



def train():

    gbdt = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.1
        , n_estimators=100
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=1
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , alpha=0.9
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )
    train_data = pd.read_csv('data/all_fea_data/all_2015-07.csv')
    y = train_data['Visits']
    y = y.reshape((y.shape[0], 1))
    del train_data['Visits']
    del train_data['Page']
    del train_data['date']
    X = train_data

    test_data = pd.read_csv("data/all_fea_data/all_2015-08.csv")
    test_y = test_data['Visits']
    test_y = test_y.reshape((test_y.shape[0], 1))
    del test_data['Visits']
    del test_data['Page']
    del test_data['date']
    test_X = test_data

    gbdt.fit(X, y)
    pred = gbdt.predict(test_X)
    total_err = 0
    for i in range(pred.shape[0]):
        err = (pred[i] - test_y[i]) / test_y[i]
        total_err += err * err
    print total_err / pred.shape[0]


if __name__ == '__main__':
    train()