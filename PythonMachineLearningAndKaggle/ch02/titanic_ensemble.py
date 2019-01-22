#!/usr/bin/env python3
# @Time    : 2018/10/4 3:34 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : titanic_ensemble.py
# @Software: PyCharm
# @Description 本例使用泰坦尼克数据对单一决策树与集成模型中的随机森立分类器以及梯度提升决策树进行比较

# 导入pandas，并且重命名为pd。
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# 从sklearn.metrics导入classification_report。
from sklearn.metrics import classification_report

# 通过互联网读取泰坦尼克乘客档案，并存储在变量titanic中。
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 人工选取pclass、age以及sex作为判别乘客是否能够生还的特征。
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

# 使用单一决策树进行模型训练以及预测分析。
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析。
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析。
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

# 输出单一决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of random forest classifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))
