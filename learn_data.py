# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 21:51'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'learn_data.py' 


import os
import pandas as pd
import numpy as np
from config import get_data_absolute_path, get_data_pickle_path
import matplotlib.pyplot as plt
import cPickle as pickle
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

def read_raw_data():
    train_1_pickle_path = '{0}/train_1.pkl'.format(get_data_pickle_path())
    if os.path.exists(train_1_pickle_path):
        train_1 = pickle.load(open(train_1_pickle_path))
    else:
        train_1_data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
        train_1 = pd.read_csv(train_1_data_path)
        pickle.dump(train_1, open(train_1_pickle_path, 'w'))
    return train_1


def read_top_8(data):
    top_8_pickle_path = '{0}/top_8.pkl'.format(get_data_pickle_path())
    if os.path.exists(top_8_pickle_path):
        top_8_visits = pickle.load(open(top_8_pickle_path))
    else:
        train_1 = read_raw_data().fillna(0.0)
        train_1['col_sum'] = train_1.apply(lambda x: x[1:].sum(), axis=1)
        top_8_visits  =train_1.nlargest(8, 'col_sum').reset_index()
        pickle.dump(top_8_visits, open(top_8_visits, 'w'))
    return top_8_visits

def do_with_top1():
    top_8_pickle_path = '{0}/top_8.pkl'.format(get_data_pickle_path())
    top_8_visits = pickle.load(open(top_8_pickle_path))
    # print top_8_visits.nlargest(2, 'col_sum').icol(0)
    # print top_8_visits.nlargest(8, 'col_sum').sort_values('col_sum', ascending=False).iloc[:,0]
    top_8_visits_sorted = top_8_visits.nlargest(8, 'col_sum').sort_values('col_sum', ascending=False)
    top_1 = top_8_visits_sorted.iloc[0,2:-1].reset_index()

    the_unusual_date = top_1[top_1.iloc[:,1] > 50000000].iloc[0, 0]
    time_series = top_1[top_1['index']<'2015-07-02'].iloc[:,1]
    model = ARIMA(np.log(time_series), order=(2, 1, 2))
    result_ARIMA=model.fit(disp=-1)
    print result_ARIMA





def read():
    data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
    raw_data = pd.read_csv(data_path).fillna(0.0)
    raw_data['col_sum'] =raw_data.apply(lambda x: x[1:].sum(), axis=1)
    max_data = raw_data[raw_data['col_sum'] == raw_data['col_sum'].max()]
    top_8_visits = raw_data.nlargest(8, 'col_sum').reset_index()

    # top_8_col = np.array(top_8_visits.columns)
    # top_8_val = np.array(top_8_visits.values)
    # for line in range(len(top_8_visits)):
    #     present_val = top_8_val[line][2:-1]
    #     label = int(top_8_val[line][0])
    #     plt.plot(present_val, label=label)
    # plt.savefig('figure.png')
    # plt.show()



    # 查看最多的那个，最多的那个id是38573
    top_1 = top_8_visits[top_8_visits['index'] == 38573]
    # 过滤没用的东西
    top_1 = top_1.iloc[:, 2:-1]

    # top_1_val = top_1.values.tolist()[0]
    # top_1_val = top_1.iloc[:, 1].values

    top_1 = top_1.T
    top_1 = top_1.reset_index()
    the_unusual_date = top_1[top_1.icol(1)>30000000].icol(0).values[0]
    print the_unusual_date

    # 过路出文件中浏览流浪比较大的几个记录， 排序在画图





if __name__ == '__main__':
    do_with_top1()
