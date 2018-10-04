#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: iris_classification.py
@time: 2017/4/27 下午1:22
"""

"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""

print (__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets


def iris_example():
    # import some data to play with
    iris = datasets.load_iris()
    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    # step size in the mesh
    h = .02

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM reqularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result in a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()


def iris_predict():
    # import some data to play with
    iris = datasets.load_iris()
    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    x = iris.data[:120]
    y = iris.target[:120]

    # step size in the mesh
    h = .02

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM reqularization parameter
    svc = svm.SVC(kernel='linear', C=C)
    svc.fit(x, y)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=C)
    rbf_svc = svm.SVC()
    rbf_svc.fit(x, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
    poly_svc.fit(x, y)
    lin_svc = svm.LinearSVC(C=C)
    lin_svc.fit(x, y)

    test_x = iris.data[60:]
    test_y = iris.target[60:]
    print(test_y)
    svc_predict_y = svc.predict(test_x)
    print("svc: ", svc_predict_y)
    rbf_svc_predict_y = rbf_svc.predict(test_x)
    print("rbf_svc: ", rbf_svc_predict_y)
    poly_svc_predict_y = poly_svc.predict(test_x)
    print("poly_svc: ", poly_svc_predict_y)
    lin_predict_y = lin_svc.predict(test_x)
    print("lin_svc: ", lin_predict_y)



if __name__ == '__main__':
    # iris_example()
    iris_predict()
