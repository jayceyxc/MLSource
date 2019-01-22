#!/usr/bin/env python3
# @Time    : 2018/10/6 2:25 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    :
# @File    : feature_selection_test.py
# @Software: PyCharm
# @Description 使用Titanic数据集，通过特征筛选的方法一步步提升决策树的性能

# 导入pandas并且更名为pd。
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
import pylab as pl

# 从互联网读取titanic数据。
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.info())

# 分离数据特征与预测目标
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

# 对缺失数据进行填充。
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

# 分割数据，依然采用25%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print("X_train type: {0}".format(type(X_train)))
"""
<class 'pandas.core.frame.DataFrame'>
"""

# 类别型特征向量化
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
print("after transform, X_train type: {0}".format(type(X_train)))
"""
压缩稀疏行矩阵
<class 'scipy.sparse.csr.csr_matrix'>
"""

# 输出处理后特征向量的维度
print(len(vec.feature_names_))

# 使用决策树模型依靠所有特征进行预测，并做性能评估
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
print('all features: {0}'.format(dt.score(X_test, y_test)))

# 导入特征筛选器，筛选前20%的特征，使用相同配置的决策树模型进行预测，并评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print('20 percent features: {0}'.format(dt.score(X_test_fs, y_test)))

# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    # print(scores)
    # print("percentile {0}, scores mean {1}".format(i, scores.mean()))
    results = np.append(results, scores.mean())
    # print(results)

print(results)
# 找到体现最佳性能的特征筛选的百分比
print(results.max())
print(np.max(results))
opt = np.where(results == results.max())[0]
print(opt)
print(type(opt))
print(percentiles)
print(np.array(percentiles))
print('Optimal number of features {0}'.format(np.array(percentiles)[opt]))

pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs, y_test))
