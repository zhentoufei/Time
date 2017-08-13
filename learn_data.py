# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 21:51'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'learn_data.py' 



import pandas as pd
import numpy as np
from config import get_data_absolute_path
def read():
    data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
    raw_data = pd.read_csv(data_path).fillna(0.0)
    raw_data['col_sum'] =raw_data.apply(lambda x: x[1:].sum(), axis=1)
    max_data = raw_data[raw_data['col_sum'] == raw_data['col_sum'].max()]

if __name__ == '__main__':
    read()
