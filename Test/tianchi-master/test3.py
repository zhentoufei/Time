# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/19 10:40'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'test3.py'

# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class arima_model:
    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxint

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic  # aic也可以
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    # 参数确定模型
    def certain_model(self, p, q):
        model = ARMA(self.data_ts, order=(p, q))
        try:
            self.properModel = model.fit(disp=-1, method='css')
            self.p = p
            self.q = q
            self.bic = self.properModel.bic
            self.predict_ts = self.properModel.predict()
            self.resid_ts = deepcopy(self.properModel.resid)
        except:
            print 'You can not fit the model with this parameter p,q, ' \
                  'please use the get_proper_model method to get the best model'

    # 预测第二日的值
    def forecast_next_day_value(self, type='day'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:  # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        predict_value = np.dot(para[1:], values)
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)


# 差分操作,d代表差分序列，比如[1,1,1]可以代表3阶差分。  [12,1]可以代表第一次差分偏移量是12，第二次差分偏移量是1
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print last_data_shift_list
        shift_ts = tmp_ts.shift(i)
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
        tmp_data = predict_value
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


import arrow


def get_date_range(start, limit, level='month', format='YYYY-MM-DD'):
    start = arrow.get(start, format)
    result = (list(map(lambda dt: dt.format(format), arrow.Arrow.range(level, start, limit=limit))))
    dateparse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    return map(dateparse2, result)


import sys

if __name__ == '__main__':
    df = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='date')
    df.index = pd.to_datetime(df.index)
    ts = df['Passengers']

    # d第一个数是移动平均阶数，第二个数是差分阶数
    # 数据预处理

    ts_log = np.log(ts)
    diffed_ts = diff_ts(ts_log, d=[12, 1])
    model = arima_model(diffed_ts)
    # model.certain_model(1, 1)
    model.get_proper_model()
    print 'bic:', model.bic, 'p:', model.p, 'q:', model.q
    print model.properModel.forecast(12)
    # print model.forecast_next_day_value(type='month')

    # predict_ts = model.properModel.predict(start='1961-01', end='1962-12')
    predict_ts = model.properModel.forecast(24)[0]
    print 'predict_ts========================'
    print predict_ts
    diff_recover_ts = predict_diff_recover(predict_ts, d=[12, 1])
    log_recover = np.exp(diff_recover_ts)
    print 'log_recover================================'
    print log_recover
    plt.plot(log_recover)
    plt.show()
    # 预测结果作图
    # ts = ts[log_recover.index]
    # plt.figure(facecolor='white')
    # log_recover.plot(color='blue', label='Predict')
    # ts.plot(color='red', label='Original')
    # plt.legend(loc='best')
    # plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - ts) ** 2) / ts.size))
    # plt.show()
