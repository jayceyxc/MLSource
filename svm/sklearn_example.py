#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: sklearn_example.py.py
@time: 2017/3/29 上午11:58
"""

from sklearn import svm

if __name__ == "__main__":
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, y)
    dec = clf.decision_function([[1]])
    dec.shape[1]   # 4 classes: 4*3/2=6
    clf.decision_function_shape = "ovr"
    dec = clf.decision_function([[1]])
    dec.shape[1]  # 4 classes
