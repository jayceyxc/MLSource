#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: trees.py.py
@time: 01/11/2017 08:37
"""

from math import log
import operator
import pickle
import tree_plotter
# from . import tree_plotter


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_entropy(data_set):
    """
    计算数据集data_set的香农熵

    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    for feature_vec in data_set:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * log(prob, 2)

    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    根据选定的特征，划分出标签值等于给定value值的子数据集

    :param data_set: 待划分的整体数据集
    :param axis: 用于划分数据集的特征下标
    :param value: 用于划分数据集的特征值
    :return:
    """
    ret_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            # reduced_feature_vec是去除选定特征剩余的特征向量
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis+1:])
            ret_data_set.append(reduced_feature_vec)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    遍历所有的特征，选择香农熵最大的特征作为分类特征

    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # 创建唯一的分类标签列表
        feature_list = [example[i] for example in data_set]
        unique_values = set(feature_list)
        new_entropy = 0.0
        # 计算每种划分方式的信息熵
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_entropy(sub_data_set)

        # 计算最好的信息增益
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set:
    :param labels:
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(data_set[0]) == 1:
        return majority_count(class_list)

    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)

    return my_tree


def classify(input_tree, feature_labels, test_vec):
    """
    使用决策树进行分类

    :param input_tree:
    :param feature_labels:
    :param test_vec:
    :return:
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_str)
    class_label = None
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                class_label = second_dict[key]

    return class_label


def store_tree(input_tree, filename):
    """
    序列化存储决策树

    :param input_tree:
    :param filename:
    :return:
    """
    with open(filename, mode='w') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    """
    读取决策树

    :param filename:
    :return:
    """
    with open(filename, mode='r') as fd:
        return pickle.load(fd)


if __name__ == '__main__':
    with open('lenses.txt', mode='r') as fd:
        lenses = [inst.strip().split('\t') for inst in fd.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = create_tree(lenses, lenses_labels)
        print(lenses_tree)
        tree_plotter.create_plot(lenses_tree)