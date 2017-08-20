# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/18 10:40'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'forcast.py'
# coding: utf-8
import requests
from datetime import datetime
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
# from pyramid.arima import ARIMA, auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from config import OPENTSDB_URL


def gen_ts(start_time, stop_time, sample=('30min', 'mean')):
    import pickle
    file_name = 'qps1.txt'
    req_url = OPENTSDB_URL

    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            res = pickle.load(f)
        ts = pd.Series(res)
    else:
        res = requests.get(url=req_url)
        if res.status_code == 200:
            ts = res.json()[0]['dps']
            ts = {datetime.fromtimestamp(float(k)): v for k, v in ts.items()}
            with open(file_name, 'wr') as f:
                pickle.dump(ts, f)
            ts = pd.Series(ts)

    if start_time and stop_time:
        ts = ts[start_time:stop_time]
    if sample:
        ts = ts.resample(sample[0], how=sample[1], closed='left')
        ts.dropna(inplace=True)
        # 数据重采样
    return ts


def plot_arima(truth, forecasts, *args):
    '''
    画图 truth is red , forecast ts is blue
    :param truth: 真实的数据
    :param forecasts:预测的数据，以预测的数据作为阈值基准
    :param args: 应该是一个长度为2的tuple.包涵阈值上限和阈值下限
    :return:
    '''
    plt.figure()
    plt.plot(truth, color='red')
    plt.plot(forecasts, color='blue')
    if args:
        for arg in args:
            plt.plot(arg, color='blue')
    plt.show(block=True)


def get_forecast(org_ts, forecast_periods, orders=(2, 1, 2), seasonal_orders=(0, 1, 1, 48), freq='30min'):
    '''
    获得预测的数据
    :param org_ts:  原始的数据
    :param forecast_periods:    预测多少个point
    :param orders:  p d q的值。p、q分别和acf和pacf相关，d是差分的阶数建议先使用auto_arima( get_suitable_orders )测试出合适的值
    :param seasonal_orders: 同上，最后一位是序列的周期
    :param freq:    表示每一个point 之间的间隔
    :return:    预测的值
    '''
    order, seasonal_order = orders, seasonal_orders
    stepwise_fit = ARIMA(order=order, seasonal_order=seasonal_order).fit(y=org_ts)
    forecast_ts = stepwise_fit.predict(n_periods=forecast_periods)

    forecasts_date_start = org_ts.index[-1] + (org_ts.index[-1] - org_ts.index[-2])
    forecast_ts = pd.Series(forecast_ts,
                            index=pd.date_range(forecasts_date_start, periods=forecast_periods, freq=freq))
    return forecast_ts


def get_suitable_orders(org_ts, periods=48):
    stepwise_fit = auto_arima(org_ts, start_p=1, start_q=1, max_p=5, max_q=5, m=periods,
                              start_P=0, seasonal=True, d=1, D=1, trace=True,
                              error_action='ignore',  # don't want to know if an order does not work
                              suppress_warnings=True,  # don't want convergence warnings
                              stepwise=True)  # set to stepwise

    stepwise_fit.summary()


def get_low_high_series(forecast_value, cal_type=('add', '10')):
    '''calculate 阈值
    :param forecast_value:  预测值
    :param cal_type: calculate 方式, 'add' mean 加减增长因子, multi mean 乘增长因子
            ex: cal_type('add', 10) ---> high = forecast_value+ 10
                cal_type('add', 0.1) ---> high = forecast_value*(1+0.1)
    :return:    阈值下限和阈值上限
    '''
    if cal_type[0] == 'add':
        low_s, high_s = ts_real.values + cal_type[1], ts_real.values - cal_type[1]
    else:
        low_s, high_s = ts_real.values * (1 + cal_type[1]), ts_real.values * (1 - cal_type[1])

    low_series = pd.Series(low, index=ts_real.index)
    high_series = pd.Series(high, index=ts_real.index)
    return low_series, high_series


ts_history = gen_ts('2017-08-02', '2017-08-08 17:00:00', sample=('20min', 'mean'))
ts_real = gen_ts('2017-08-07 16:30:00', '2017-08-09 12:30:00', sample=('20min', 'mean'))

forecast_data = get_forecast(ts_history, 50, orders=(1, 1, 1), seasonal_orders=(0, 1, 1, 72), freq='20min')
low, high = get_low_high_series(forecast_data, cal_type=('add', 20))

plot_arima(ts_real, forecast_data, low, high)

print ts_real, forecast_data
