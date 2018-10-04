#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: data_normalization.py
@time: 2017/5/3 上午9:32
"""

### 数据规范化

import pandas as pd
import numpy as np


def main():
    # 参数初始化
    datafile = 'data/normalization_data.xls'
    # 读取数据
    data = pd.read_excel(datafile, header=None)
    print data

    # 最小-最大规范化
    print (data - data.min()) / (data.max() - data.min())
    # 零-均值规范化
    print (data - data.mean())/data.std()
    # 小数定标规范化
    print data / 10**np.ceil(np.log10(data.abs().max()))


if __name__ == '__main__':
    main()