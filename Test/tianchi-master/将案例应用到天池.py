# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_csv('/Users/wangtuntun/Desktop/023406156015ef87f99521f3b343f71f', parse_dates='Day', index_col='Day',
                   date_parser=dateparse)
# data=pd.read_csv('/Users/wangtuntun/Desktop/AirPassengers.csv')
# print data.head()
ts = data['#Play']
# print ts.head(10)
# plt.plot(ts)
# plt.show()
# test_stationarity(ts)
# plt.show()
ts_log = np.log(ts)
# plt.plot(ts_log)
# plt.show()
moving_avg = pd.rolling_mean(ts_log, 365)
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

'''
#确定参数
lag_acf=acf(ts_log_diff,nlags=100)
lag_pacf=pacf(ts_log_diff,nlags=100,method='ols')
#q的获取:ACF图中曲线第一次穿过上置信区间.这里q取0
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')#lowwer置信区间
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')#upper置信区间
plt.title('Autocorrelation Function')
#p的获取:PACF图中曲线第一次穿过上置信区间.这里p取0
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
'''

# order p d q
model = ARIMA(ts_log, order=(0, 1, 0))
result_ARIMA = model.fit(disp=-1)
predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA, color='red')
plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
plt.show()

Day,  # Play
20150301, 487
20150302, 502
20150303, 542
20150304, 585
20150305, 599
20150306, 587
20150307, 569
20150308, 519
20150309, 570
20150310, 602
20150311, 643
20150312, 673
20150313, 644
20150314, 568
20150315, 569
20150316, 508
20150317, 604
20150318, 638
20150319, 595
20150320, 596
20150321, 572
20150322, 702
20150323, 559
20150324, 551
20150325, 609
20150326, 560
20150327, 625
20150328, 682
20150329, 647
20150330, 764
20150331, 594
20150401, 709
20150402, 646
20150403, 656
20150404, 546
20150405, 582
20150406, 481
20150407, 531
20150408, 648
20150409, 616
20150410, 705
20150411, 620
20150412, 567
20150413, 560
20150414, 559
20150415, 646
20150416, 625
20150417, 703
20150418, 578
20150419, 564
20150420, 590
20150421, 730
20150422, 689
20150423, 601
20150424, 804
20150425, 641
20150426, 566
20150427, 602
20150428, 721
20150429, 733
20150430, 592
20150501, 616
20150502, 823
20150503, 721
20150504, 557
20150505, 781
20150506, 612
20150507, 590
20150508, 649
20150509, 787
20150510, 616
20150511, 699
20150512, 760
20150513, 783
20150514, 771
20150515, 628
20150516, 584
20150517, 640
20150518, 840
20150519, 718
20150520, 644
20150521, 654
20150522, 766
20150523, 788
20150524, 817
20150525, 691
20150526, 777
20150527, 746
20150528, 724
20150529, 637
20150530, 688
20150531, 772
20150601, 740
20150602, 607
20150603, 732
20150604, 788
20150605, 771
20150606, 620
20150607, 624
20150608, 578
20150609, 591
20150610, 927
20150611, 734
20150612, 709
20150613, 750
20150614, 842
20150615, 686
20150616, 1479
20150617, 769
20150618, 782
20150619, 801
20150620, 792
20150621, 693
20150622, 720
20150623, 680
20150624, 598
20150625, 785
20150626, 861
20150627, 836
20150628, 675
20150629, 785
20150630, 761
20150701, 705
20150702, 752
20150703, 700
20150704, 663
20150705, 703
20150706, 803
20150707, 821
20150708, 681
20150709, 708
20150710, 667
20150711, 680
20150712, 822
20150713, 727
20150714, 775
20150715, 924
20150716, 799
20150717, 697
20150718, 813
20150719, 791
20150720, 734
20150721, 778
20150722, 769
20150723, 798
20150724, 1066
20150725, 848
20150726, 748
20150727, 786
20150728, 818
20150729, 680
20150730, 1080
20150731, 762
20150801, 738
20150802, 624
20150803, 776
20150804, 808
20150805, 818
20150806, 665
20150807, 673
20150808, 618
20150809, 711
20150810, 650
20150811, 817
20150812, 657
20150813, 674
20150814, 714
20150815, 787
20150816, 942
20150817, 632
20150818, 673
20150819, 700
20150820, 764
20150821, 937
20150822, 638
20150823, 785
20150824, 766
20150825, 714
20150826, 870
20150827, 775
20150828, 803
20150829, 785
20150830, 682
