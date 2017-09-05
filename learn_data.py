# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 21:51'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'learn_data.py'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from config import get_data_absolute_path, get_data_pickle_path
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from ARIMA_tools import proper_model, get_date_range, predict_diff_recover, diff_ts
from statsmodels.tsa.stattools import adfuller


def read_raw_data():
    '''
    读取train_1.csv的文件并且保存到本地pickle文件
    :return: 
    '''
    train_1_pickle_path = '{0}/train_1.pkl'.format(get_data_pickle_path())
    if os.path.exists(train_1_pickle_path):
        train_1 = pickle.load(open(train_1_pickle_path))
    else:
        train_1_data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
        train_1 = pd.read_csv(train_1_data_path)
        pickle.dump(train_1, open(train_1_pickle_path, 'w'))
    return train_1


def read_top_8():
    top_8_pickle_path = '{0}/top_8.pkl'.format(get_data_pickle_path())
    if os.path.exists(top_8_pickle_path):
        top_8_visits = pickle.load(open(top_8_pickle_path))
    else:
        train_1 = read_raw_data().fillna(0.0)
        train_1['col_sum'] = train_1.apply(lambda x: x[1:].sum(), axis=1)
        top_8_visits = train_1.nlargest(8, 'col_sum').reset_index()
        pickle.dump(top_8_visits, open(top_8_pickle_path, 'w'))
    return top_8_visits


def do_with_top1():
    '''
    找到流量访问最多的那个网页并观察网页的访问情况，重新流量预测
    :return: 
    '''
    date_list = ['2016-07-21', '2016-07-22', '2016-07-23', '2016-07-24', '2016-07-25', '2016-07-26'
        , '2016-07-27', '2016-07-28', '2016-07-29', '2016-07-30', '2016-07-31', '2016-08-01'
        , '2016-08-02', '2016-08-03', '2016-08-04', '2016-08-05', '2016-08-06', '2016-08-07'
        , '2016-08-08', '2016-08-09', '2016-08-10', '2016-08-11', '2016-08-12', '2016-08-13'
        , '2016-08-14', '2016-08-15', '2016-08-16']
    top_8_pickle_path = '{0}/top_8.pkl'.format(get_data_pickle_path())
    top_8_visits = pickle.load(open(top_8_pickle_path))
    # print top_8_visits.nlargest(2, 'col_sum').icol(0)
    # print top_8_visits.nlargest(8, 'col_sum').sort_values('col_sum', ascending=False).iloc[:,0]
    top_8_visits_sorted = top_8_visits.nlargest(8, 'col_sum').sort_values('col_sum', ascending=False)
    top_1 = top_8_visits_sorted.iloc[0, 2:-1].reset_index()

    the_unusual_date = top_1[top_1.iloc[:, 1] > 50000000]
    # print the_unusual_date
    # print the_unusual_date.count()

    the_train_data = top_1[top_1['index'] < '2016-07-18']
    the_train_data.to_csv('tmp.csv')
    forecast_ARIMA = do_with_top1_pre_direct()
    top_1.rename(columns={'index': 'date', 0: 'visits'}, inplace=True)
    top_1_upper = top_1[top_1['date']< '2016-07-18']
    top_1_lower = top_1[top_1['date']>'2016-09-09']
    top_1_middle = forecast_ARIMA
    frames = [top_1_upper, top_1_middle, top_1_lower]
    top_1 = pd.concat(frames, ignore_index=True)
    print top_1.head(30)
    plt.plot(top_1['visits'])
    plt.show()
    print forecast_ARIMA.head()


def do_with_top1_decompose():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    the_train_data = pd.read_csv('tmp.csv', parse_dates=['index'], index_col='index', date_parser=dateparse)
    ts = the_train_data.iloc[:, 1]
    ts_log = np.log(ts)
    decompostion = seasonal_decompose(ts_log)
    trend = decompostion.trend
    seasonal = decompostion.seasonal
    residual = decompostion.resid
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('pic/decompostion.png')
    plt.show()


def do_with_top1_pre_direct():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    the_train_data = pd.read_csv('tmp.csv', parse_dates=['index'], index_col='index', date_parser=dateparse)
    ts = the_train_data.iloc[:, 1]

    ts_log = np.log(ts)
    # 原来的数据检验，但是不满足1%的检验
    # adf_test(the_train_data_log, 'the_train_data_log')

    # 做一阶差分, 可以满足1%的检验
    # ts_log_diff = ts_log - ts_log.shift(periods=4)
    diff = [1, 1]
    ts_log_diff2 = diff_ts(ts_log, diff)
    ts_log_diff2.dropna(inplace=True)
    # adf_test(ts_log_diff, 'ts_log_diff')

    # 寻找最佳的参数
    # p, q, results_ARIMA = proper_model(ts_log_diff2, 10)


    model = ARIMA(ts_log, order=(5, 2, 7))
    results_ARIMA = model.fit(disp=-1)

    # plt.plot(ts_log_diff2)
    # plt.plot(results_ARIMA.fittedvalues, color='black')
    # plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff2)**2))
    # plt.show()


    predict_index = get_date_range('2016-07-19', 62, format='YYYY-MM-DD')
    forecast_ARIMA = results_ARIMA.forecast(62)[0]
    forecast_ARIMA = np.exp(forecast_ARIMA)
    forecast_ARIMA = pd.Series(forecast_ARIMA, copy=True, index=predict_index)
    forecast_ARIMA = forecast_ARIMA.reset_index()
    forecast_ARIMA.rename(columns={'index': 'date', 0: 'visits'}, inplace=True)
    print forecast_ARIMA.head()
    return forecast_ARIMA


def adf_test(timeseries, name):
    rolling_statistics(timeseries, name)  # 绘图
    print 'Results of Augment Dickey-Fuller Test:'
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


def rolling_statistics(timeseries, name):
    # Determing rolling statistics
    rolmean = pd.Series.rolling(timeseries, window=12).mean()
    rolstd = pd.Series.rolling(timeseries, window=12).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('pic/' + name + '.png')


def read():
    data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
    raw_data = pd.read_csv(data_path).fillna(0.0)
    raw_data['col_sum'] = raw_data.apply(lambda x: x[1:].sum(), axis=1)
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
    the_unusual_date = top_1[top_1.icol(1) > 30000000].icol(0).values[0]
    print the_unusual_date

    # 过路出文件中浏览流浪比较大的几个记录， 排序在画图


if __name__ == '__main__':
    do_with_top1()
    # do_with_top1_pre_direct()
    # do_with_top1_decompose()
