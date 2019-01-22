#!/usr/bin/env python3
# @Time    : 2018/10/4 5:06 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : digits_kmeans_cluster.py
# @Software: PyCharm
# @Description 使用KMeans算法对数字进行聚类

# 分别导入numpy、matplotlib以及pandas，用于数学运算、作图以及数据分析。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 从sklearn.cluster中导入KMeans模型。
from sklearn.cluster import KMeans
# 从sklearn导入度量函数库metrics。
from sklearn import metrics

# 使用pandas分别读取训练数据与测试数据集。
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 从训练与测试数据集上都分离出64维度的像素特征与1维度的数字目标。
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 初始化KMeans模型，并设置聚类中心数量为10。
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)

# 逐条判断每个测试图像所属的聚类中心。
y_pred = kmeans.predict(X_test)

print(metrics.adjusted_rand_score(y_test, y_pred))





