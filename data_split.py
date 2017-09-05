# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 17:10'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'data_split.py'

import pandas as pd
from config import *
import os
import re


def get_lannguage(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[0:2]
    return 'na'


def read_train():
    train_data_path = os.path.join(get_data_absolute_path(), "train_1.csv")
    df_train = pd.read_csv(train_data_path).fillna(0.0)
    all_col = df_train.columns

    df_train['access'] = df_train['Page'].apply(lambda x: x.split('_')[-2])
    df_train['agent'] = df_train['Page'].apply(lambda x: x.split('_')[-1])
    df_train['title'] = df_train['Page'].apply(lambda x: x.split('.wikipedia')[0])

    # ================================================
    # train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    for month in range(7, 13):
        present_month = '2015-{0}'.format(str(month).zfill(2))
        tmp_list = []
        for ele in all_col:
            if present_month in ele:
                tmp_list.append(ele)

        path = os.path.join(get_data_absolute_path(), 'split_data/Page_{0}.csv'.format(present_month))
        col_list = ['Page','title', 'access', 'agent'] + tmp_list
        train_flattened = pd.melt(df_train[col_list], id_vars=['Page','title', 'access', 'agent'],
                                  var_name='date', value_name='visits')
        # train_flattened.drop(train_flattened.columns[[0]], axis=1, inplace=True)  # 删除第一列

        train_flattened['lang'] = train_flattened['Page'].map(get_lannguage)  # 获取语言类型
        train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
        train_flattened['month'] = (train_flattened.date.dt.month).astype(float)
        train_flattened['day'] = (train_flattened.date.dt.day).astype(float)
        train_flattened['weekday'] = (train_flattened.date.dt.dayofweek).astype(float)
        train_flattened['is_weenkend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

        del train_flattened['date']
        del df_train['Page']

        train_flattened = pd.get_dummies(train_flattened,
                                         columns=['month', 'day', 'weekday', 'is_weenkend', 'access', 'agent'],
                                         prefix=['month', 'day', 'weekday', 'is_weenkend', 'access', 'agent'])

        train_flattened.to_csv(path, encoding='utf-8')
        print("finish: {0}".format(path))

    for month in range(1, 13):
        present_month = '2016-{0}'.format(str(month).zfill(2))
        tmp_list = []
        for ele in all_col:
            if present_month in ele:
                tmp_list.append(ele)

        path = os.path.join(get_data_absolute_path(), 'split_data/Page_{0}.csv'.format(present_month))
        col_list = ['Page','title', 'access', 'agent'] + tmp_list
        train_flattened = pd.melt(df_train[col_list], id_vars=['Page','title', 'access', 'agent'],
                                  var_name='date', value_name='visits')
        # train_flattened.drop(train_flattened.columns[[0]], axis=1, inplace=True)  # 删除第一列

        train_flattened['year'] = train_flattened['date'].apply(lambda x: x.split('-')[0])
        train_flattened['month'] = train_flattened['date'].apply(lambda x: x.split('-')[1])
        train_flattened['day'] = train_flattened['date'].apply(lambda x: x.split('-')[2])

        train_flattened['lang'] = train_flattened.Page.map(get_lannguage)  # 获取语言类型
        train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
        train_flattened['month'] = (train_flattened.date.dt.month).astype(float)
        train_flattened['day'] = (train_flattened.date.dt.day).astype(float)
        train_flattened['weekday'] = (train_flattened.date.dt.dayofweek).astype(float)
        train_flattened['is_weenkend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

        del train_flattened['date']
        del df_train['Page']
        train_flattened = pd.get_dummies(train_flattened,
                                         columns=['month', 'day', 'weekday', 'is_weenkend', 'access', 'agent'],
                                         prefix=['month', 'day', 'weekday', 'is_weenkend', 'access', 'agent'])

        train_flattened.to_csv(path, encoding='utf-8')
        print("finish: {0}".format(path))


if __name__ == '__main__':
    read_train()
