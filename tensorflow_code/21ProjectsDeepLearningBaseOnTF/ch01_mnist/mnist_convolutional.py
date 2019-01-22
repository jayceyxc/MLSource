#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: mnist_convolutional.py
@time: 2018/9/8 10:48 PM
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    """
    返回一个给定形状的变量，并自动以截断正态分布初始化
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    返回一个给定形状的变量，初始化时所有值是0.1
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    卷积层
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    池化层
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


# 第一层卷积层


# 读入数据
mnist = input_data.read_data_sets("/Users/yuxuecheng/TF_data/MNIST_data/", one_hot=True)
# x为训练图像的占位符，y_为训练图像标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将单张图片从784维向量还原成28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层，输出为1024维的向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用Dropout, keep_prob是一个占位符，训练时为0.5，测试时为1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 把1024维的向量转换成10维，对应10个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 不采用先Softmax在计算交叉熵的方法
# 而是用直接计算
cross_entrophy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 定义训练步骤
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entrophy)

# 定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建Session，对变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练20000步
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 每100步报告一次在验证集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, trainning accuracy: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束后报告在测试集上的准确率
print("test accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


