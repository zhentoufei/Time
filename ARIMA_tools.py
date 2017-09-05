# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/20 14:03'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'ARIMA.py'

import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARMA, _arima_model
from config import get_data_pickle_path, get_data_absolute_path

import arrow


# 差分操作,d代表差分序列，比如[1,1,1]可以代表3阶差分。  [12,1]可以代表第一次差分偏移量是12，第二次差分偏移量是1
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list  # 这个序列在恢复过程中需要用到
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)  #
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts


# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i - 1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i - 1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i - 1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data  # return np.exp(tmp_data)也可以return到最原始，tmp_data是对原始数据取对数的结果


def get_date_range(start, limit, level='day', format='YYYY-MM-DD'):
    start = arrow.get(start, format)
    result = (list(map(lambda dt: dt.format(format), arrow.Arrow.range(level, start, limit=limit))))
    dateparse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    return map(dateparse2, result)


def proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxint
    best_model = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            print('present p: ', p, '  q: ', q)
            model = ARMA(ts_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            # print bic, best_bic
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    print('best_p: ', best_p)
    print('best_q: ', best_q)
    return best_model


if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv('AirPassengers.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)
