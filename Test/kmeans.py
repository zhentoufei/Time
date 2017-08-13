# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/12 10:57'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'kmeans.py'

import numpy as np
import matplotlib.pyplot as plt
# 随机生成一个实数，范围在（0.5,1.5）之间
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
# hstack拼接操作
X = np.hstack((cluster1, cluster2)).T
plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:, 0], X[:, 1], 'k.')
plt.show()