#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm
@file: svmMLiA.py
@time: 2018/5/29 11:04
"""

import random
import numpy as np


def load_data_set(filename):
    """
    从文件中加载数据

    :param filename:
    :return:
    """
    data_mat = []
    label_mat = []

    with open(filename, mode="r") as fd:
        for line in fd.readlines():
            line_arr = line.strip().split("\t")
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))

    return data_mat, label_mat


def select_j_rand(i, m):
    """
    在0到m之间随机选择一个不等于i的整数

    :param i: 第一个alpha的小标
    :param m: 所有alpha的数目
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j


def clip_alpha(aj, H, L):
    """
    用于调整大于H或小于L的alpha值

    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H

    if L > aj:
        aj = L

    return aj


def simple_smo(data_mat_in, class_labels, C, tolerance, max_iter):
    """

    :param data_mat_in: 训练数据集
    :param class_labels: 训练数据集对应的标签
    :param C: 常数C
    :param tolerance: 容错率
    :param max_iter: 参数没有更新的情况下的最大循环次数

    :return: 常数b，参数矩阵alphas
    """
    # 转换成Numpy矩阵，简化后面的数学操作
    data_matrix = np.mat(data_mat_in)
    # class_labels被转成向量并且进行转置，使我们有一个列向量，而不是一个列表。
    label_matrix = np.mat(class_labels).transpose()

    b = 0
    m, n = np.shape(data_matrix)

    # alphas是一个 m * 1 的列向量矩阵
    # In[38]: alphas1 = np.mat(np.zeros((10, 1)))
    #
    # In[39]: alphas1
    # Out[39]:
    # matrix([[0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.],
    #         [0.]])
    #
    # In[40]: alphas1.shape
    # Out[40]: (10, 1)
    #
    # In[41]:
    alphas = np.mat(np.zeros((m, 1)))

    # 在没有任何alpha改变的情况下遍历数据集的次数，当该变量达到输入值max_iter时，函数结束运行并退出
    iter = 0
    while iter < max_iter:
        # alpha_pairs_changed记录alpha是否已经进行优化
        alpha_pairs_changed = 0
        for i in range(m):
            # fXi就是我们对第i个数据预测的类别,np.multiply(alphas, label_matrix).T是 1*m 的列向量
            # In[51]: np.multiply(alphas, label_matrix).shape
            # Out[51]: (100, 1)
            #
            # In[52]: np.multiply(alphas, label_matrix).T.shape
            # Out[52]: (1, 100)
            #
            # In[59]: (data_matrix * data_matrix[10, :].T).shape
            # Out[59]: (100, 1)
            fXi = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b

            # 预测结果与真实结果的误差Ei
            Ei = fXi - float(label_matrix[i])
            if ((label_matrix[i] * Ei < -tolerance) and (alphas[i] < C)) or ((label_matrix[i] * Ei > tolerance) and (alphas[i] > 0)):
                # 判断是否还能对alphas[i]进行优化
                j = select_j_rand(i, m)
                # fXj就是我们对第j个数据预测的类别
                fXj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                # 预测结果与真实结果的误差Ej
                Ej = fXj - float(label_matrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # L和H是alpha参数的最小值和最大值
                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L==H")
                    continue

                # Eta is the optimal amount to change alpha[j] eta是修改alpha[j]的最优修改量
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= label_matrix[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                # 判断alphas[j]的修改量是否太小
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue

                alphas[i] += label_matrix[j] * label_matrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alphaIold) * \
                     data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alphaJold) * \
                     data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_matrix[i] * (alphas[i] - alphaIold) * \
                     data_matrix[i, :] * data_matrix[j, :].T - \
                     label_matrix[j] * (alphas[j] - alphaJold) * \
                     data_matrix[j, :] * data_matrix[j, :].T

                if (alphas[i] > 0) and (C > alphas[i]):
                    b = b1
                elif (alphas[j] > 0) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0

        print("iteration number: %d" % iter)

    return b, alphas


class optStruct:
    def __init__(self, data_matrix_in, class_labels, C, tolerance):
        self.X = data_matrix_in
        self.label_matrix = class_labels
        self.C = C
        self.tol = tolerance
        self.m = np.shape(data_matrix_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))    # Error cache


def calcEk(oS, k):
    """
    计算指定alpha的E值并返回
    :param oS:
    :param k:
    :return:
    """
    fXk = float(np.multiply(oS.alphas, oS.label_matrix).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.label_matrix[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = select_j_rand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
