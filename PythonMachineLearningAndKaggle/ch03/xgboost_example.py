#!/usr/bin/env python3
# @Time    : 2018/10/7 11:46 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : xgboost_example.py
# @Software: PyCharm
# @Description 对比随机决策森林以及XGBoost模型对泰坦尼克号上的乘客是否生还的预测能力

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 通过URL地址来下载Titanic数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 选取pclass、age以及sex作为训练特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这样可以在保证顺利训练模型的同时，尽可能不影响预测任务。
X['age'].fillna(X['age'].mean(), inplace=True)

# 对原始数据进行分割，25%的乘客数据用于测试。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 对类别型特征进行转化，成为特征向量。
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用随机森林分类器进行集成模型的训练以及预测分析。
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of Random Forest Classifier on testing set: {0}'.format(rfc.score(X_test, y_test)))
print(classification_report(rfc_y_pred, y_test))

# 采用默认配置的XGBoost模型对相同的测试集进行预测
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
xgbc_y_pred = xgbc.predict(X_test)

# 输出XGBoost分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set: {0}'.format(xgbc.score(X_test, y_test)))
print(classification_report(xgbc_y_pred, y_test))

