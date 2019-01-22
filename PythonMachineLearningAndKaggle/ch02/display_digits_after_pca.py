#!/usr/bin/env python3
# @Time    : 2018/10/4 5:21 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : display_digits_after_pca.py
# @Software: PyCharm
# @Description 显示手写体数字图片经PCA压缩后的二维空间分布

# 导入pandas用于数据读取和处理。
import pandas as pd
# 导入numpy工具包。
import numpy as np

# 从互联网读入手写体图片识别任务的训练数据，存储在变量digits_train中。
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                           header=None)

# 从互联网读入手写体图片识别任务的测试数据，存储在变量digits_test中。
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                          header=None)

# 分割训练数据的特征向量和标记。
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 从sklearn.decomposition导入PCA。
from sklearn.decomposition import PCA

# 初始化一个可以将高维度特征向量（64维）压缩至2个维度的PCA。
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布。
from matplotlib import pyplot as plt


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


plot_pca_scatter()
