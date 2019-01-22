#!/usr/bin/env python3
# @Time    : 2018/10/4 2:50 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : iris_knn.py
# @Software: PyCharm
# @Description 使用K近邻方法进行iris数据分类

# 从sklearn.datasets 导入 iris数据加载器。
from sklearn.datasets import load_iris
# 从sklearn.model_selection 导入 train_test_split。
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里选择导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier，即K近邻分类器。
from sklearn.neighbors import KNeighborsClassifier
# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report

# 使用加载器读取数据并且存入变量iris。
iris = load_iris()
# 查验数据规模。
print(iris.data.shape)
# 查看数据说明。对于一名机器学习的实践者来讲，这是一个好习惯。
print(iris.DESCR)

# 从使用train_test_split，利用随机种子random_state采样25%的数据作为测试集。
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

# 对训练和测试的特征数据进行标准化。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用K近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict中。
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评。
print('Accuracy os K-Nearest Neighbor Classifier is {0}'.format(knc.score(X_test, y_test)))

print(classification_report(y_test, y_predict, target_names=iris.target_names))
