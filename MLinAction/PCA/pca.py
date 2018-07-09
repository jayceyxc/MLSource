#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: pca.py
@time: 01/12/2017 08:55
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_data_set(file_name, delim='\t'):
    with open(file_name, mode='r') as fd:
        string_arr = [line.strip().split(delim) for line in fd.readlines()]
        data_arr = [list(map(float, line)) for line in string_arr]
        return np.mat(data_arr)


def pca(data_mat, top_n_feature=9999999):
    """

    :param data_mat:
    :param top_n_feature:
    :return:
    """
    mean_values = np.mean(data_mat, axis=0)  # 计算平均值
    mean_removed = data_mat - mean_values
    cov_mat = np.cov(mean_removed, rowvar=0)  # 计算协方差矩阵
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_mat))
    eig_values_index = np.argsort(eig_values)
    eig_values_index = eig_values_index[: -(top_n_feature+1) : -1]
    reg_eig_vectors = eig_vectors[:, eig_values_index]
    low_d_data_mat = mean_removed * reg_eig_vectors
    recon_mat = (low_d_data_mat * reg_eig_vectors.T) + mean_values

    return low_d_data_mat, recon_mat


def test_n_feature(top_n_feature=9999999):
    """

    :param top_n_feature:
    :return:
    """
    data_mat = load_data_set('testSet.txt')
    low_d_mat, recon_mat = pca(data_mat, 1)
    print(np.shape(low_d_mat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def replace_nan_with_mean():
    data_mat = load_data_set('secom.data', ' ')
    num_feature = np.shape(data_mat)[1]
    for i in range(num_feature):
        mean_value = np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:, i].A))[0], i])

        data_mat[np.nonzero(np.isnan(data_mat[:, i].A))[0], i] = mean_value

    return data_mat


if __name__ == '__main__':
    test_n_feature(1)