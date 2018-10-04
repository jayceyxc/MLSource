#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: lagrange_newton_interp.py
@time: 2017/5/3 上午8:53
"""

### 拉格朗日插值代码
import pandas as pd
from scipy import interpolate

input_file = 'data/catering_sale.xls'
out_file = 'tmp/sales.xls'

data = pd.read_excel(input_file)
# 过滤异常值，将其变为空值
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None


# 自定义列向量插值函数
# s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def poly_interp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))] # 取数
    y = y[y.notnull()] # 剔除空值
    return interpolate.lagrange(y.index, list(y))(n)

# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            # 如果为空即插值
            data[i][j] = poly_interp_column(data[i], j)

data.to_excel(out_file) # 输出结果，写入文件