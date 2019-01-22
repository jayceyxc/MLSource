#!/usr/bin/env python3
# @Time    : 2018/10/7 5:07 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : random_forest_example.py
# @Software: PyCharm
# @Description 使用RandomForestClassifier对Titanic罹难患者预测竞赛

# 导入pandas方便数据读取和预处理
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# 从流行工具包xgboost导入XGBClassifier用于处理分类预测问题
from xgboost import XGBClassifier

# 分别对训练和测试数据从本地进行读取
train = pd.read_csv('../../Datasets/Titanic/train.csv')
test = pd.read_csv('../../Datasets/Titanic/test.csv')

# 先分别输出训练与测试数据的基本信息。这是一个好习惯，可以对数据的规模、各个特征的数据类型以及是否有缺失等，有一个总体的了解

print(train.info())
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
"""
print(train.describe())
print(test.info())
"""
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
"""
print(test.describe())

# 按照我们之前对Titanic事件的经验，人工选取对预测有效的特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']

# 通过我们之前对数据的总体观察，得知Embarked特征存在缺失值，需要补完
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())

# 对于Embarked这种类别型的特征，我们使用出现频率最高的特征值来填充，这也是相对可以减少引入误差的一种填充方法
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# 而对于Age这种数值类型的特征，我们习惯使用求平均值或中位数来填充缺失值，也是相对可以减少引入误差的一种填充方法。
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

print('After preprocessing')
# 重新对处理后的训练和测试数据进行查验
print(X_train.info())
print(X_train.describe())
print(X_test.info())
print(X_test.describe())

# 接下来便是采用DictVectorizer对特征向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 使用默认配置初始化RandomForestClassifier
rfc = RandomForestClassifier()
# 使用默认配置初始化XGBClassifier
xgbc = XGBClassifier()

# 使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier以及XGBClassifier进行性能评估，并获得平均分类准确性的得分
print(cross_val_score(rfc, X_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, X_train, y_train, cv=5).mean())

# 使用默认配置初始化RandomForestClassifier进行预测操作
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submissions = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
# 将默认配置的RandomForestClassifier对测试数据的预测结果存储在文件rfc_submission.csv中
rfc_submissions.to_csv('../../Datasets/Titanic/rfc_submission.csv', index=False)

# 使用默认配置初始化XGBClassifier进行预测操作
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
# 将默认配置的XGBClassifier对测试数据的预测结果存储在文件xgbc_submission.csv中
xgbc_submissions = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submissions.to_csv('../../Datasets/Titanic/xgbc_submission.csv', index=False)

# 使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提高XGBClassifier的预测性能
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)

# 查验优化之后的XGBClassifier的超参数配置以及交叉验证的准确性
print(gs.best_score_)
print(gs.best_params_)

# 使用经过优化超参数配置的XGBClassifier对测试数据的预测结果存储在文件xgbc_best_submission.csv
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submissions = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submissions.to_csv('../../Datasets/Titanic/xgbc_best_submission.csv', index=False)





