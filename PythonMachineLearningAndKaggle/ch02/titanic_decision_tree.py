#!/usr/bin/env python3
# @Time    : 2018/10/4 3:04 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : titanic_decision_tree.py
# @Software: PyCharm
# @Description 使用决策树算法对泰坦尼克事件中人员生还状态

# 导入pandas用于数据分析。
import pandas as pd
# 数据分割。
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
# 从sklearn.tree中导入决策树分类器。
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import graphviz


# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据。
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 观察一下前几行数据，可以发现，数据种类各异，数值型、类别型，甚至还有缺失数据。
print(titanic.head())
"""
	row.names	pclass	survived	name	age	embarked	home.dest	room	ticket	boat	sex
0	1	1st	1	Allen, Miss Elisabeth Walton	29.0000	Southampton	St Louis, MO	B-5	24160 L221	2	female
1	2	1st	0	Allison, Miss Helen Loraine	2.0000	Southampton	Montreal, PQ / Chesterville, ON	C26	NaN	NaN	female
2	3	1st	0	Allison, Mr Hudson Joshua Creighton	30.0000	Southampton	Montreal, PQ / Chesterville, ON	C26	NaN	(135)	male
3	4	1st	0	Allison, Mrs Hudson J.C. (Bessie Waldo Daniels)	25.0000	Southampton	Montreal, PQ / Chesterville, ON	C26	NaN	NaN	female
4	5	1st	1	Allison, Master Hudson Trevor	0.9167	Southampton	Montreal, PQ / Chesterville, ON	C22	NaN	11	male
"""
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，查看数据的统计特性。
print(titanic.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 11 columns):
row.names    1313 non-null int64
pclass       1313 non-null object
survived     1313 non-null int64
name         1313 non-null object
age          633 non-null float64
embarked     821 non-null object
home.dest    754 non-null object
room         77 non-null object
ticket       69 non-null object
boat         347 non-null object
sex          1313 non-null object
dtypes: float64(1), int64(2), object(8)
memory usage: 112.9+ KB
"""
# 机器学习有一个不太被初学者重视，并且耗时，但是十分重要的一环，特征的选择，这个需要基于一些背景知识。
# 根据我们对这场事故的了解，sex, age, pclass这些都很有可能是决定幸免与否的关键因素。
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对当前选择的特征进行探查。
print(X.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 3 columns):
pclass    1313 non-null object
age       633 non-null float64
sex       1313 non-null object
dtypes: float64(1), object(2)
memory usage: 30.9+ KB
"""

# 借由上面的输出，我们设计如下几个数据处理的任务：
# 1) age这个数据列，只有633个，需要补完。
# 2) sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替。

# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
X['age'].fillna(X['age'].mean(), inplace=True)

# 对当前选择的特征进行探查。
print(X.info())
"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1313 entries, 0 to 1312
Data columns (total 3 columns):
pclass    1313 non-null object
age       1313 non-null float64
sex       1313 non-null object
dtypes: float64(1), object(2)
memory usage: 41.0+ KB
"""


# 由此得知，age特征得到了补完。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)

# 我们使用scikit-learn.feature_extraction中的特征转换器
vec = DictVectorizer(sparse=False)

# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
print(X_train)
"""
['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
[[31.19418104  0.          0.          1.          0.          1.        ]
 [31.19418104  1.          0.          0.          1.          0.        ]
 [31.19418104  0.          0.          1.          0.          1.        ]
 ...
 [12.          0.          1.          0.          1.          0.        ]
 [18.          0.          1.          0.          0.          1.        ]
 [31.19418104  0.          0.          1.          1.          0.        ]]
"""

# 同样需要对测试数据的特征进行转换。
X_test = vec.transform(X_test.to_dict(orient='record'))
print(X_test)
"""
[[18.          0.          1.          0.          0.          1.        ]
 [48.          1.          0.          0.          1.          0.        ]
 [31.19418104  1.          0.          0.          0.          1.        ]
 ...
 [20.          0.          0.          1.          0.          1.        ]
 [54.          1.          0.          0.          0.          1.        ]
 [31.19418104  0.          0.          1.          1.          0.        ]]
"""
# 使用默认配置初始化决策树分类器。
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据进行模型学习。
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测。
y_predict = dtc.predict(X_test)


# 从sklearn.metrics导入classification_report。
from sklearn.metrics import classification_report
# 输出预测准确性。
print(dtc.score(X_test, y_test))
# 输出更加详细的分类性能。
print(classification_report(y_predict, y_test, target_names=['died', 'survived']))

dot_data = tree.export_graphviz(dtc, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("titanic")
