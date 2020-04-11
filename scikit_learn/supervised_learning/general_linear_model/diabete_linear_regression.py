#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: diabete_linear_regression.py
@time: 2018/9/3 9:55 AM
"""

from sklearn import datasets
from sklearn import linear_model
import numpy as np

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_Y_train)
print(regr.coef_)

# The mean square error
print(np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test)**2))

print(regr.score(diabetes_X_test, diabetes_Y_test))


