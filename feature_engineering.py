# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/11 19:01'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'feature_engineering.py'

import pandas as pd
import numpy as np
import pandas.tseries.holiday as hol
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans


# US holidays
us_cal = hol.USFederalHolidayCalendar()
dr = pd.date_range(start='2015-07-01', end='2017-08-01')
us_holidays = us_cal.holidays(start=dr.min(), end=dr.max())



def read_train_1():
    date_list = {'2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                 '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06',
                 '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12'}
    for date in date_list:
        file_name = 'data/split_data/Page_{0}.csv'.format(date)
        print("processing {0}".format(file_name))

        train_data = pd.read_csv(file_name).fillna(0.0)

        list_data = np.array(train_data.iloc[:, 2:])
        kmeans = KMeans(n_clusters=4, random_state=0).fit(list_data)
        labels = kmeans.labels_
        labels = labels.tolist()
        train_data['labels_in_month'] = labels
        train_data['access'] = train_data['Page'].apply(lambda x: x.split('_')[-2])
        train_data['agent'] = train_data['Page'].apply(lambda x: x.split('_')[-1])

        column_list = list(train_data)[1:]  # 第一个是原来的索引列
        column_list.insert(0, column_list.pop(column_list.index('labels_in_month')))
        column_list.insert(0, column_list.pop(column_list.index('labels_in_all')))
        column_list.insert(0, column_list.pop(column_list.index('access')))
        column_list.insert(0, column_list.pop(column_list.index('agent')))
        column_list.insert(0, column_list.pop(column_list.index('Page')))
        train_data = train_data.ix[:, column_list]

        train_data = pd.melt(
            train_data[list(train_data.columns[5:]) + ['Page', 'agent', 'access', 'labels_in_all', 'labels_in_month']],
            id_vars=['Page', 'agent', 'access', 'labels_in_all', 'labels_in_month'], var_name='date',
            value_name='Visits')

        print('获取语言类型 ', date)
        train_data['lang'] = train_data.Page.map(get_lannguage)  # 获取语言类型
        train_data['date'] = train_data['date'].astype('datetime64[ns]')
        train_data['weekday'] = (train_data.date.dt.dayofweek).astype(float)
        train_data['is_weenkend'] = ((train_data.date.dt.dayofweek) // 5 == 1).astype(float)

        print('不同国家的节假日 ', date)
        train_data['holiday'] = train_data.date.isin(us_holidays).astype(float)

        print('one-hot编码 ', date)
        train_data = pd.get_dummies(train_data,
                                    columns=['agent', 'access', 'labels_in_all', 'labels_in_month', 'lang', 'weekday'],
                                    prefix=['agent', 'access', 'labels_in_all', 'labels_in_month', 'lang', 'weekday'])

        save_path = 'data/all_fea_data/all_{0}.csv'.format(date)
        print("save at {0}".format(save_path))
        train_data.to_csv(save_path)


# 2016-12-31
# 2015-07-01

def make_offline_train_set(data):
    start_date = ''
    end_date = ''
    return


def make_offline_test_set(data):
    pass


def make_online_train_set(data):
    pass


def get_lannguage(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[0:2]
    return 'na'


def plot_language(train):
    lang_sets = {}
    lang_sets['en'] = train[train.lang == 'en'].iloc[:, 0:-1]
    lang_sets['ja'] = train[train.lang == 'ja'].iloc[:, 0:-1]
    lang_sets['de'] = train[train.lang == 'de'].iloc[:, 0:-1]
    lang_sets['na'] = train[train.lang == 'na'].iloc[:, 0:-1]
    lang_sets['fr'] = train[train.lang == 'fr'].iloc[:, 0:-1]
    lang_sets['zh'] = train[train.lang == 'zh'].iloc[:, 0:-1]
    lang_sets['ru'] = train[train.lang == 'ru'].iloc[:, 0:-1]
    lang_sets['es'] = train[train.lang == 'es'].iloc[:, 0:-1]

    sums = {}
    for key in lang_sets:
        sums[key] = lang_sets[key].iloc[:, 1:].sum(axis=0) / lang_sets[key].shape[0]

    print sums['en'].shape
    days = [r for r in range(sums['en'].shape[0])]

    fig = plt.figure(1, figsize=[10, 10])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title('Pages in Different Languages')
    labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
              'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
              'ru': 'Russian', 'es': 'Spanish'
              }

    for key in sums:
        plt.plot(days, sums[key], label=labels[key])

    plt.legend()
    plt.savefig('output/figure.png')


if __name__ == '__main__':
    # read_test()
    train = read_train_1()
    # get_language_count(train)
    # print get_lannguage('aa.wikipedia.org')
