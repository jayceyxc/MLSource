#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_classification_navie_bayes.py
@time: 2017/8/4 23:50
"""

import os

import jieba
from jieba import analyse
from sklearn.naive_bayes import MultinomialNB

from utility import text_utility


def train_navie_bayes():
    """
    train the navie bayes model
    
    :return: None
    """
    train_url_cat, train_url_content = text_utility.get_documents(current_path="data", pattern="train_*.xlsx")
    vocabulary = dict()
    matrix = text_utility.my_extract(list(train_url_content.values()), vocabulary)
    features = dict((value, key) for key, value in vocabulary.items())
    clf = MultinomialNB().fit(matrix, list(train_url_cat.values()))

    return clf, vocabulary, features


def test_navie_bayes(test_file, classifier, vocabulary):
    """
    test the navie bayes model
    
    :param: test_file the file name of test file than contain the test content
    :param: classifier the navie bayes model that has been trained
    :param: vocabulary the vocabulary the classifier use
    :return: None
    """
    right_count = 0
    total_count = 0
    test_url_cat, test_url_content = text_utility.get_documents(current_path="data", pattern="test_*.xlsx")
    url_list = list(test_url_cat.keys())
    content_list = list(test_url_content.values())
    cat_list = list(test_url_cat.values())
    for i in range(0, len(url_list), 1):
        url = url_list[i]
        content = content_list[i]
        expected_cat = cat_list[i]
        content_matrix = text_utility.get_content_tfidf(content, vocabulary)
        cat = classifier.predict(content_matrix)
        total_count += 1
        if cat[0] == expected_cat:
            right_count += 1

        print(u"\t".join([url, cat[0], expected_cat]))

    print("right count: {0}, total count: {1}, accuracy: {2}".format(right_count, total_count, float(right_count) / total_count))


if __name__ == "__main__":
    jieba.load_userdict("dict" + os.sep + "user.dict")
    analyse.set_stop_words("dict" + os.sep +"stop_words.txt")
    clf, vocabulary, features = train_navie_bayes()
    test_file_name = "data" + os.sep + "test_1.xslx"
    test_navie_bayes(test_file=test_file_name, classifier=clf, vocabulary=vocabulary)