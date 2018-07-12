#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_classification_knn.py
@time: 2017/8/8 14:11
"""

import os
import traceback

import jieba
from jieba import analyse
from sklearn.neighbors import KNeighborsClassifier

from utility import text_utility


def train_knn(n_neighbors=15, weights='uniform'):
    """
    train the knn model

    :return: None
    """
    train_url_cat, train_url_content = text_utility.get_documents(current_path="data", pattern="train_*.xlsx")
    vocabulary = dict()
    matrix = text_utility.my_extract(train_url_content.values(), vocabulary)
    features = dict((value, key) for key, value in vocabulary.items())
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(matrix, train_url_cat.values())

    return clf, vocabulary, features


def test_knn(test_file, classifier, vocabulary):
    """
     test the navie bayes model

     :param: test_file the file name of test file than contain the test content
     :param: classifier the navie bayes model that has been trained
     :param: vocabulary the vocabulary the classifier use
     :return: None
     """
    with open(test_file, mode='r') as fd:
        first = True
        for line in fd:
            if first:
                first = False
                continue

            try:
                indptr = [0]
                indices = []
                data = []
                line = line.strip()
                segs = line.split(',')
                if len(segs) != 6:
                    continue
                url, title, keywords, desc, a_content, p_content = line.split(',')
                content = " ".join([title, keywords, desc, a_content, p_content])
                content_matrix = text_utility.get_content_tfidf(content, vocabulary)

                cat = classifier.predict(content_matrix)

                print(u"\t".join([url, cat[0]]))
            except UnicodeDecodeError as ude:
                traceback.print_exc()
                continue


if __name__ == "__main__":
    jieba.load_userdict("dict" + os.sep + "user.dict")
    analyse.set_stop_words("dict" + os.sep +"stop_words.txt")
    for weight in ["uniform", "distance"]:
        for neighbor in range(4, 15, 1):
            print("ngighbor: {0}, weight: {1}".format(neighbor, weight))
            clf, vocabulary, features = train_knn(n_neighbors=neighbor, weights=weight)
            test_file_name = "data" + os.sep + "test.txt"
            test_knn(test_file=test_file_name, classifier=clf, vocabulary=vocabulary)