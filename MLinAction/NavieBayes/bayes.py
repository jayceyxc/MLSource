#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: bayes.py
@time: 02/11/2017 08:48
"""

import numpy as np
import re
import random


def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    # 1代表侮辱性文字，0代表正常言论
    class_vec = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)

    return list(vocab_set)


def set_of_word_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocalbulary!" % word)

    return return_vec


def bag_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print("the word: %s is not in my vocalbulary!" % word)

    return return_vec


def train_navie_bayes_0(train_matrix, train_category):
    """
    训练Navie Bayes模型

    :param train_matrix: 训练数据矩阵
    :param train_category: 训练数据分类
    :return: p0_vec 每个单词出现在类别0中的概率
             p1_vec 每个单词出现在类别1中的概率
             p_abusive
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category)/float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    """
    当计算乘积p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时，由于大部分因子都非常小，所以程序会下溢出或者得不到正确的答案
    一种解决的办法就是对乘积取自然对数，ln(a*b) = ln(a) + ln(b)
    """
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)

    return p0_vec, p1_vec, p_abusive


def classify_navie_bayes(vec_to_classify, p0_vec, p1_vec, p_class1):
    """
    用Navie Bayes方法进行判断分类

    :param vec_to_classify: 待分类的向量
    :param p0_vec: 分类0的概率向量，即每个单词属于分类0的概率组成的向量，向量长度为词汇表长度
    :param p1_vec: 分类1的概率向量，即每个单词属于分类1的概率组成的向量，向量长度为词汇表长度
    :param p_class1: 分类1的全概率，即分类1的文档数除以文档总数得到的概率
    :return: 待分类向量所属的分类
    """

    """
    这里对分类1的概率取对数，是因为在训练函数中，对于单词属于某个分类的概率取了对数
    按照对数公式: ln(a * b) = ln(a) + ln(b)
    则贝叶斯公式的分母p(w|ci)*p(ci) 取对数后就变成了log(p(w|ci)) + log(p(ci))
    """
    p1 = sum(vec_to_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec_to_classify * p0_vec) + np.log(1.0 - p_class1)

    if p1 > p0:
        return 1
    else:
        return 0


def testing_navie_bayes():
    """
    测试贝叶斯模型

    :return:
    """
    posts_list, class_list = load_data_set()
    my_vocab_list = create_vocab_list(posts_list)
    train_matrix = []
    for posting_doc in posts_list:
        train_matrix.append(set_of_word_to_vec(my_vocab_list, posting_doc))

    p0_vec, p1_vec, p_ab = train_navie_bayes_0(np.array(train_matrix), np.array(class_list))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_word_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_navie_bayes(this_doc, p0_vec, p1_vec, p_ab))

    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_word_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_navie_bayes(this_doc, p0_vec, p1_vec, p_ab))


"""
spam email testing example
"""


def text_parse(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        try:
            word_list = text_parse(open('email/spam/%d.txt' % i).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)
        except UnicodeDecodeError as ude:
            print("file: " + 'email/spam/%d.txt' % i)
            print(ude.reason)

        try:
            word_list = text_parse(open('email/ham/%d.txt' % i).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        except UnicodeDecodeError as ude:
            print("file: " + 'email/spam/%d.txt' % i)
            print(ude.reason)

    vocab_list = create_vocab_list(doc_list)
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_matrix = []
    train_classes = []

    for doc_index in training_set:
        train_matrix.append(set_of_word_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    p0_vec, p1_vec, p_spam = train_navie_bayes_0(np.array(train_matrix), np.array(train_classes))
    error_count = 0

    for doc_index in test_set:
        word_vector = set_of_word_to_vec(vocab_list, doc_list[doc_index])
        if classify_navie_bayes(np.array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            print("error classify index: {0}".format(doc_index))
            print(doc_list[doc_index])
            error_count += 1

    print('the error rate is: ', float(error_count) / len(test_set))
