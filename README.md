# Time

###查看一下流量最大的几个网页

```python
data_path = '{0}/train_1.csv'.format(get_data_absolute_path())
raw_data = pd.read_csv(data_path).fillna(0.0)
raw_data['col_sum'] =raw_data.apply(lambda x: x[1:].sum(), axis=1)
max_data = raw_data[raw_data['col_sum'] == raw_data['col_sum'].max()]
colu_sum = raw_data['col_sum'].sort_values()
```

得到的结果是如下：

在10次方数量级的：38573

在9次方数量级的：9774，74114，139119，39180116196，99322，10403