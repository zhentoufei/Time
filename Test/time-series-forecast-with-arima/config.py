# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/8/18 10:39'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'config.py'
OPENTSDB_URL = 'http://tsdb.zhxfei.com/api/query?start=14d-ago&m=avg:rate:nginx_req_total{endpoint=zhxfei.com,domain=newapi.zhxfei.com,port=80}'

import sys

if __name__ == '__main__':
    arr = sys.stdin.readline().strip()
    if arr:
        arr = arr.split()
        index = 0
        max_area = 0
        while (index < len(arr) - 1):
            tmp = 2 * min(arr[index], arr[index + 1])
            if tmp > max_area:
                max_area = tmp
            index += 1
        print max_area
