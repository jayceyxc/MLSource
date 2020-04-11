#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: iris_knn.py
@time: 2018/9/2 9:58 PM
"""

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

print(np.unique(iris_Y))

np.random.seed(0)
indices = np.random.permutation(len(iris_Y))
iris_X_train = iris_X[indices[:-10]]
iris_Y_train = iris_Y[indices[:-10]]

iris_X_test = iris_X[indices[-10:]]
iris_Y_test = iris_Y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_Y_train)
print(knn.n_neighbors)
iris_Y_predicted = knn.predict(iris_X_test)
print(iris_Y_predicted)
print(iris_Y_test)
print(np.equal(iris_Y_predicted, iris_Y_test))



