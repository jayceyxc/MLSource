#!/usr/bin/env python3
# @Time    : 2018/10/9 3:45 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : neural_network.py
# @Software: PyCharm
# @Description 神经网络算法预测销量高低

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# 参数初始化
input_file = '../data/sales_data.xls'
# 导入数据
data = pd.read_excel(input_file, index_col=u'序号')

# 数据是类别标签，要将它转换为数据
# 用1来表示"好"、"是"、"高"这三个属性，用0表示"坏"、"否"、"低"
data[data == u'好'] = 1
data[data == u'高'] = 1
data[data == u'是'] = 1
data[data != 1] = 0
x = data.iloc[:, :3].as_matrix().astype(int)
y = data.iloc[:, 3].as_matrix().astype(int)

model = Sequential()
model.add(Dense(3, 10))
# 用relu函数作为激活函数，能够大幅度提高准确度
model.add(Activation('relu'))
model.add(Dense(10, 1))
# 由于是0-1输出，用sigmod函数作为激活函数
model.add(Activation('sigmod'))
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
model.fit(x, y, nb_epoch=1000, batch_size=10)
yp = model.predict_classes(x).reshape(len(y))
print(yp)





