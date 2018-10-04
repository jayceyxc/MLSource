#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: data_discretization.py
@time: 2017/5/3 下午12:41
"""

import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt


def cluster_plot(data, d, k):
    """
    自定义作图函数来显示聚类结果
    :param data
    :param d: 
    :param k: 
    :return: 
    """
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')

    plt.ylim(-0.5, k-0.5)
    return plt


def data_discretization():
    data_file = 'data/discretization_data.xls'
    data = pd.read_excel(data_file)
    data = data[u'肝气郁结证型系数'].copy()

    k = 4
    # 等宽离散化，各个类依次命名为0，1，2，3
    d1 = pd.cut(data, k, labels=range(k))

    # 等频率离散化
    w = [1.0*i/k for i in range(k + 1)]
    # 使用describe函数自动计算分位数
    w = data.describe(percentiles = w)[4:4+k+1]
    w[0] = w[0] * (1-1e-10)
    d2 = pd.cut(data, w, labels=range(k))

    # 建立模型，n_jobs是并行数，一般等于CPU数较好
    kmodel = cluster.KMeans(n_clusters=k, n_jobs=4)
    # 训练模型
    kmodel.fit(data.reshape((len(data), 1)))
    # 输出聚类中心，并且排序（默认是随机排序的）
    c = pd.DataFrame(kmodel.cluster_centers_).sort(0)

    # 相邻两点求中点，作为边界点
    w = pd.rolling_mean(c, 2).iloc[1:]
    # 把首末边界点加上
    w = [0] + list(w[0]) + [data.max()]
    d3 = pd.cut(data, w, labels=range(k))
    cluster_plot(data, d1, k).show()
    cluster_plot(data, d2, k).show()
    cluster_plot(data, d3, k).show()


if __name__ == '__main__':
    data_discretization()