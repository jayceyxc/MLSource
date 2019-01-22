#!/usr/bin/env python3
# @Time    : 2018/10/4 12:29 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : digits_recognition_svm.py
# @Software: PyCharm
# @Description 利用SVM进行手写数字识别

from sklearn.datasets import load_digits
# 使用sklearn.model_selection里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
# 从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC。
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中。
digits = load_digits()
# 检视数据规模和特征维度。
print(digits.data.shape)

# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print(Y_train.shape)
print(Y_test.shape)

# 对训练和测试的特征数据进行标准化。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 初始化线性假设的支持向量机分类器LinearSVC。
linear_svc = LinearSVC()
# 进行模型训练
linear_svc.fit(X_train, Y_train)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中。
y_predict = linear_svc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评。
print('The Accuracy of Linear SVC is {0}'.format(linear_svc.score(X_test, Y_test)))

# 使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
print(classification_report(Y_test, y_predict, target_names=digits.target_names.astype(str)))



