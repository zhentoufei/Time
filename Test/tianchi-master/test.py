# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/15 21:50'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'test.py' 

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
import test_stationarity




# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='date')
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
ts = df['Passengers']  # 生成pd.Series对象
# 查看数据格式
# print ts.head()
# print ts['1949']

ts_log = np.log(ts)
test_stationarity.draw_ts(ts_log)
test_stationarity.draw_trend(ts_log, 12)