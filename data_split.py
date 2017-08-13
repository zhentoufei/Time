# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/13 17:10'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'data_split.py'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
train_data_path = 'input/train_1.csv'


def read_train():
    df_train = pd.read_csv(train_data_path).fillna(0.0)
    all_col = df_train.columns

    # ================================================
    # 添加聚类标签
    print '添加聚类标签'
    tmp_page = df_train['Page']
    df_train.drop('Page', axis=1, inplace=True)
    list_data = np.array(df_train)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(list_data)
    labels = kmeans.labels_
    labels = labels.tolist()
    df_train['labels_in_all'] = labels
    df_train['Page'] = tmp_page
    df_train.to_csv('data/raw_data.csv')
    for month in range(7, 13):
        present_month = '2015-{0}'.format(str(month).zfill(2))
        tmp_list = []
        for ele in all_col:
            if present_month in ele:
                tmp_list.append(ele)

        path = 'data/split_data/Page_{0}.csv'.format(present_month)
        col_list = ['Page', 'labels_in_all'] + tmp_list
        df_train[col_list].to_csv(path)
        print "finish: {0}".format(path)

    for month in range(1, 13):
        present_month = '2016-{0}'.format(str(month).zfill(2))
        tmp_list = []
        for ele in all_col:
            if present_month in ele:
                tmp_list.append(ele)

        path = 'data/split_data/Page_{0}.csv'.format(present_month)
        col_list = ['Page', 'labels_in_all'] + tmp_list
        df_train[col_list].to_csv(path)
        print "finish: {0}".format(path)


if __name__ == '__main__':
    read_train()
