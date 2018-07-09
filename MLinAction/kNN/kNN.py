#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: kNN.py
@time: 31/10/2017 13:13
"""

import operator
import os

import matplotlib.pyplot as plt
import numpy as np


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    k近邻算法，该程序使用欧式距离公式

    :param in_x: 用于分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 标签向量，标签向量的元素数目和data_set的行数相同
    :param k: 用于选择最近邻居的数目
    :return: 返回输入向量inX对应的标签
    """
    # 计算距离
    # 获取x轴维度的长度
    data_set_size = data_set.shape[0]
    # 将in_x根据data_set_size进行平铺，然后减去data_set，得到和data_set中每个样本的差值
    """
    In [20]: in_x = np.array([1,2])

    In [21]: np.tile(in_x,(group.shape[0], 1))
    Out[21]:
    array([[1, 2],
           [1, 2],
           [1, 2],
           [1, 2]])
    """
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 计算差值的平方
    sq_diff_mat = diff_mat ** 2
    # 计算差值平方的和
    sq_distance = sq_diff_mat.sum(axis=1)
    # 计算差值平方和的开方值
    distances = sq_distance ** 0.5
    # 按照距离按从小到大的进行排序，返回距离从小到大的样本的下标
    sorted_dist_indices = distances.argsort()

    class_count = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 根据索引下标获取对应的标签
        vote_i_label = labels[sorted_dist_indices[i]]
        # 统计各个标签的数量
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # 排序，根据标签数量降序排列
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 获取第一个标签
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    将文本记录转换成Numpy矩阵的解析程序
    :param filename:
    :return:
    """
    with open(filename) as fr:
        # 得到文件行数
        array_of_ines = fr.readlines()
        number_of_lines = len(array_of_ines)
        # 创建返回的Numpy矩阵
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_of_ines:
            # 解析文件数据到列表
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append(int(list_from_line[-1]))
            index += 1

        return return_mat, class_label_vector


def print_scatter_diagram(dating_data_mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
    plt.show()


def print_scatter_diagram_with_labels(dating_data_mat, dating_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels),
               15.0 * np.array(dating_labels))
    plt.show()


def auto_norm(data_set):
    """
    对数据集data_set进行归一化
    :param data_set:
    :return:
    """
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = max_values - min_values
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_values, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_values


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0

    for i in range(num_test_vecs):
        """
        In [29]: type(norm_mat)
        Out[29]: numpy.ndarray
        
        In [30]: norm_mat.shape
        Out[30]: (1000, 3)
        
        In [31]: type(dating_labels)
        Out[31]: list
        """
        # norm_mat[i, :] 选取第i行的数据，","前是行的范围，后是列的范围
        # norm_mat[num_test_vecs:m, :] 选取第num_test_vecs到m行的数据
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer: %d" % (classifier_result, dating_labels[i]))

        if classifier_result != dating_labels[i]:
            error_count += 1.0

    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    """
    根据输入的参数对约会人进行分类
    :return:
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, mim_values = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - mim_values) / ranges, norm_mat, dating_labels, 3)
    print("you will probably like this person: ", result_list[classifier_result - 1])


def img2vector(filename):
    return_vector = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vector[0, 32 * i + j] = int(line_str[j])

    return return_vector


def hand_writing_class_test():
    """
    使用k-近邻算法识别手写数字
    :return:
    """
    hw_labels = []
    training_file_list = os.listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)

    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_num_str))

        if classifier_result != class_num_str:
            error_count += 1.0

    print('\nthe total number of errors is: %d' % error_count)
    print('\nthe total error rate is: %f' % (error_count / float(m_test)))


if __name__ == '__main__':
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
