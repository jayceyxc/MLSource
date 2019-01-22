#!/usr/bin/env python3
# @Time    : 2018/10/9 2:49 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : logistic_regression.py
# @Software: PyCharm
# @Description 逻辑回归代码

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression

# 参数初始化
file_name = '../data/bankloan.xls'
data = pd.read_excel(file_name)
print(data.describe())
X = data.iloc[:, :8].as_matrix()
y = data.iloc[:, 8].as_matrix()

# 建立随机逻辑回归模型，筛选变量
rlr = RandomizedLogisticRegression()
# 训练模型
rlr.fit(X, y)
# 获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
print(rlr.get_support())
# print(X.columns[rlr.get_support()])
# print(','.join(X.columns[rlr.get_support()]))
# print(u'通过随机逻辑回归模型筛选特征结束。')
# print(u'有效特征为：{0}'.format(','.join(X.columns[rlr.get_support()])))
# # 筛选好特征
# x = data[data.columns[rlr.get_support()]].as_matrix()

# 建立逻辑回归模型
lr = LogisticRegression()
# 用筛选后的特征数据来训练模型
lr.fit(X, y)
print(u'逻辑回归模型训练结束。')
# 给出模型的平均正确率，本例为81.4%
print(u'模型的平均正确率为：{0}'.format(lr.score(X, y)))


