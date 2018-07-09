#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: logRegres.py
@time: 03/11/2017 08:48
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random


def plot_best_fit(weights):
    # weights = wei.getA()
    data_matrix, label_matrix = load_data_set()
    data_arr = np.array(data_matrix)
    n = np.shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    for i in range(n):
        if int(label_matrix[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def load_data_set():
    data_matrix = []
    label_matrix = []

    with open('testSet.txt', mode='r') as fd:
        for line in fd:
            line_arr = line.strip().split()
            data_matrix.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_matrix.append(int(line_arr[2]))

    return data_matrix, label_matrix


def sigmod(in_x):
    return 1.0/(1 + np.exp(-in_x))


def grad_ascent(data_matrix_in, class_labels):
    """
    gradient ascent 梯度上升算法

    :param data_matrix_in:
    :param class_labels:
    :return:
    """
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmod(data_matrix * weights)
        error = (label_matrix - h)
        # print(error)
        # print("\r\n")
        weights = weights + alpha * data_matrix.transpose() * error
        # print(weights)
        # print("\r\n")

    h = sigmod(data_matrix * weights)
    print(h)
    print("\r\n")
    error = (label_matrix - h)
    print(error)
    print("\r\n")
    return weights


def stoc_grad_ascent0(data_matrix, class_labels):
    """
    stochastic gradient ascent 随机梯度上升算法
    :param data_matrix: 输入
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmod(np.sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    """
    改进版的随机梯度上升算法。主要是两个改进的地方：
    1、alpha在每次迭代的时候都会调整，缓解数据波动或者高频波动。另外，虽然
       alpha会随着迭代次数不断减小，但永远不会到0，这是因为调整还有一个常数项
       这样做的原因是保证在多次迭代之后新数据仍然具有一定的影响。
    2、通过随机选取样本来更新回归系统，这种方法可以减少周期性的波动。

    :param data_matrix: 训练数据
    :param class_labels: 训练数据的标签结果
    :param num_iter: 迭代次数
    :return:
    """
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            # alpha每次迭代时需要调整
            alpha = 4/(1.0+j+i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmod(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])

    return weights


# 预测病马死亡率的例子
def classify_vector(in_x, weights):
    prob = sigmod(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))

        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    # print train_weights
    error_count = 0
    number_test_vec = 0.0
    for line in fr_test.readlines():
        # print line
        number_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))

        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / number_test_vec)

    print("the error rate of this test is: %f" % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()

    print("after %d iterations the average error rate is: %f" % (num_tests, error_sum/float(num_tests)))

