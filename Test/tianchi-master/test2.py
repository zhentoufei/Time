# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/17 13:28'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'test2.py'

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
import test_stationarity
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARMA, _arima_model
import statsmodels.api as sm


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
        print last_data_shift_list
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


def adf_test(timeseries, name):
    rolling_statistics(timeseries, name)  # 绘图
    print 'Results of Augment Dickey-Fuller Test:'
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


def acf_pacf_plot(ts_log_diff):
    sm.graphics.tsa.plot_acf(ts_log_diff, lags=40)  # ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff, lags=40)  # ARIMA,p
    plt.savefig('pic/acf_pacf_plot.png')


def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxint
    best_model = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
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
    print 'best_p', best_p
    print 'best_q', best_q
    return best_p, best_q, best_model


def do_something():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv('AirPassengers.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

    print data.head()
    ts = data['Passengers']

    ts_log = np.log(ts)

    # 平滑处理
    moving_avg = pd.Series.rolling(ts_log, 12).mean()
    # plt.plot(ts_log)
    # plt.plot(moving_avg, color='red')
    # plt.show()

    # 做差,均值和原始值的差
    ts_log_moving_avg_diff = ts_log - moving_avg
    adf_test(ts_log_moving_avg_diff.dropna(), 'ts_log_moving_avg_diff')
    plt.close()

    # 指数加权移动平均
    # 前面移动平均数需要指定window, 并且对所有的数一视同仁；
    # 这里采用指数加权移动平均方法，会对当前的数据加大权重，对过去的数据减小权重。
    # halflife半衰期，用来定义衰减量。其他参数, 如跨度span和质心com也可以用来定义衰减。
    expwighted_avg = pd.ewma(ts_log, halflife=12)
    plt.plot(ts_log)
    plt.plot(expwighted_avg, color='red')
    plt.savefig('pic/expwighted_avg.png')
    ts_log_ewma_diff = ts_log - expwighted_avg

    adf_test(ts_log_ewma_diff, 'ts_log_ewma_diff')

    # Test Statistic - 3.601262
    # p-value                          0.005737
    # #Lags Used                      13.000000
    # Number of Observations Used    130.000000
    # Critical Value (5%)             -2.884042
    # Critical Value (1%)             -3.481682
    # Critical Value (10%)            -2.578770
    # 可以发现，经过指数移动平均后，再做差的结果，已经能够通过1%显著性水平检验了。


    # 步长为1的一阶差分
    plt.close()
    ts_log_diff = ts_log - ts_log.shift(periods=1)
    ts_log_diff.dropna(inplace=True)
    # adf_test(ts_log_diff, 'ts_log_diff')

    # 只通过了10%的显著性检验


    # 我们继续进行二阶差分
    # 一阶差分：Y(k)=X(k+1)-X(k)
    # 二阶差分：Y(k)的一阶差分Z(k)=Y(k+1)-Y(k)=X(k+2)-2*X(k+1)+X(k)为此函数的二阶差分
    ts_log_diff = ts_log - ts_log.shift(periods=1)
    ts_log_diff2 = ts_log_diff - ts_log_diff.shift(periods=1)
    plt.plot(ts_log_diff2)
    ts_log_diff2.dropna(inplace=True)
    adf_test(ts_log_diff2, 'ts_log_diff2')
    # 可以看到，二阶差分，p值非常小，小于1%，检验统计量也明显小于%1的临界值。因此认定为很平稳

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

    # 对残差进行ADF检验
    # 可以发现序列非常平稳
    plt.close()
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    adf_test(ts_log_decompose, 'ts_log_decompose')
    # 对残差进行ADF检验，可以发现序列非常平稳。




    # 前面我们对数据进行ADF检验，判断序列是否平稳，这里我们使用自相关图和偏自相关图对数据平稳性再次进行验证，一阶差分如下图：
    acf_pacf_plot(ts_log_diff)  # 调用一阶差分


# 注意这里面使用的ts_log_diff是经过合适阶数的差分之后的数据，上文中提到ARIMA该开源库，不支持3阶以上的#差分。所以我们需要提前将数据差分好再传入

import arrow
def get_date_range(start, limit, level='month',format='YYYY-MM-DD'):
    start = arrow.get(start, format)
    result=(list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level, start, limit=limit))))
    dateparse2 = lambda dates:pd.datetime.strptime(dates,'%Y-%m')
    return map(dateparse2, result)

if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv('AirPassengers.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

    ts = data['Passengers']
    print ts
    d = [1]  # 定义差分序列
    ts_log = np.log(ts)
    ts_log_diff = diff_ts(ts_log, d)
    # res = _proper_model(ts_log_diff, 9)  # 对一阶差分求最优p和q
    model = ARIMA(ts_log, order=(10,1,7)) # 第二个参数表示的是一阶差分
    results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='black')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
    plt.show()
    print '==============================='
    new_index = get_date_range('1961-01', 12, format='YYYY-MM')
    # forecast = predict_diff_recover(results_ARIMA.forecast(12)[0],d)
    predic = pd.Series(np.exp(results_ARIMA.forecast(12)[0]), copy=True, index=new_index)
    print predic
    # predict_diff_recover
    plt.plot(predic, color='red')
    plt.plot(ts)
    plt.show()




    # plt.plot(results_ARIMA.forecast(12)[0], color='red')
    # plt.plot(np.exp(results_ARIMA.fittedvalues), color='black')
    # plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
    # print model.predict(pd.datetime.strptime('1961-01', '%Y-%m'), 24)
    #  p和q: 8, 9
    # print res
