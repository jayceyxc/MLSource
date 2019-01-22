#!/usr/bin/env python3
# @Time    : 2018/10/6 5:49 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : predict_pizza_price.py
# @Software: PyCharm
# @Description 使用线性回归模型在披萨训练样本上进行拟合

# 从sklearn.linear_model中导入LinearRegression。
from sklearn.linear_model import LinearRegression
# 从sklearn.preproessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt


# 输入训练样本的特征以及目标值，分别存储在变量X_train与y_train之中。
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 使用默认配置初始化线性回归模型。
regressor = LinearRegression()
# 直接以披萨的直径作为特征训练模型。
regressor.fit(X_train, y_train)

# 在x轴上从0至25均匀采样100个数据点。
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
print(xx)

# 以上述100个数据点作为基准，预测回归直线。
yy = regressor.predict(xx)

# 对回归预测到的直线进行作图。
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()
# 输出线性回归模型在训练样本上的R-squared值。
print('The R-squared value of Linear Regressor performing on the training data is {0}'
      .format(regressor.score(X_train, y_train)))

# 使用PolynominalFeatures(degree=2)映射出2次多项式特征，存储在变量X_train_poly2中。
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
print(X_train_poly2)
"""
[[  1.   6.  36.]
 [  1.   8.  64.]
 [  1.  10. 100.]
 [  1.  14. 196.]
 [  1.  18. 324.]]
"""

# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型。
regressor_poly2 = LinearRegression()

# 对2次多项式回归模型进行训练。
regressor_poly2.fit(X_train_poly2, y_train)

# 从新映射绘图用x轴采样数据。
xx_poly2 = poly2.transform(xx)

# 使用2次多项式回归模型对应x轴采样数据进行回归预测。
yy_poly2 = regressor_poly2.predict(xx_poly2)

# 分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图。
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1, plt2])
plt.show()

# 输出2次多项式回归模型在训练样本上的R-squared值。
print('The R-squared value of Polynominal Regressor (Degree=2) performing on the training data is {0}'
      .format(regressor_poly2.score(X_train_poly2, y_train)))

# 初始化4次多项式特征生成器。
poly4 = PolynomialFeatures(degree=4)

X_train_poly4 = poly4.fit_transform(X_train)
print(X_train_poly4)
"""
[[1.00000e+00 6.00000e+00 3.60000e+01 2.16000e+02 1.29600e+03]
 [1.00000e+00 8.00000e+00 6.40000e+01 5.12000e+02 4.09600e+03]
 [1.00000e+00 1.00000e+01 1.00000e+02 1.00000e+03 1.00000e+04]
 [1.00000e+00 1.40000e+01 1.96000e+02 2.74400e+03 3.84160e+04]
 [1.00000e+00 1.80000e+01 3.24000e+02 5.83200e+03 1.04976e+05]]
"""

# 使用默认配置初始化4次多项式回归器。
regressor_poly4 = LinearRegression()
# 对4次多项式回归模型进行训练。
regressor_poly4.fit(X_train_poly4, y_train)

# 从新映射绘图用x轴采样数据。
xx_poly4 = poly4.transform(xx)
# 使用4次多项式回归模型对应x轴采样数据进行回归预测。
yy_poly4 = regressor_poly4.predict(xx_poly4)

# 分别对训练数据点、线性回归直线、2次多项式以及4次多项式回归曲线进行作图。
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()

print('The R-squared value of Polynominal Regressor (Degree=4) performing on the training data is {0}'
      .format(regressor_poly4.score(X_train_poly4, y_train)))

# 准备测试数据。
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 使用测试数据对线性回归模型的性能进行评估。
print(regressor.score(X_test, y_test))
print(regressor.coef_)
"""
[[0.9762931]]
"""
print(regressor.intercept_)
"""
[1.96551724]
"""

# 使用测试数据对2次多项式回归模型的性能进行评估。
X_test_poly2 = poly2.transform(X_test)
print(regressor_poly2.score(X_test_poly2, y_test))
print(regressor_poly2.coef_)
"""
[[ 0.          2.95615672 -0.08202292]]
"""
print(regressor_poly2.intercept_)
"""
[-8.39765458]
"""

# 使用测试数据对4次多项式回归模型的性能进行评估。
X_test_poly4 = poly4.transform(X_test)
print(regressor_poly4.score(X_test_poly4, y_test))
print(regressor_poly4.coef_)
"""
[[ 0.00000000e+00 -2.51739583e+01  3.68906250e+00 -2.12760417e-01
   4.29687500e-03]]
"""
print(regressor_poly4.intercept_)
"""
[65.625]
"""
print(np.sum(regressor_poly4.coef_ ** 2))
