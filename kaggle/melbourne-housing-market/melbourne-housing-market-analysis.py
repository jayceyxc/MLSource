#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: melbourne-housing-market-analysis.py
@time: 2017/5/5 上午9:08
"""

import pandas as pd
import matplotlib.pyplot as plt


def visualization():
    plt.figure()
    data = pd.read_csv("/Users/yuxuecheng/data/input/Melbourne_housing_extra_data.csv", parse_dates=[7])
    data.plot()
    plt.show()


if __name__ == '__main__':
    visualization()
