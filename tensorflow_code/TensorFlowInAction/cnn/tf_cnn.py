#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: tf_cnn.py
@time: 14/12/2017 14:33
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    """
    使用截断的正态分布噪声，标准差设为0.1
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


def bias_variable(shape):
    """
    给偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）。
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


def conv2d(x, W):
    """
    tf.nn.conv2d是TensorFlow中的2维卷积函数，strides代表卷积模板移动的步长，都是1代表会不遗漏的划过图片的每一个点。padding代表
    边界的处理方式，这里的SAME代表个边界加上padding让卷积的输出和输入保持同样的尺寸
    :param x: 输入
    :param W: 卷积的参数，比如[5,5,1,32]:前面两个数字代表卷积核的尺寸；第三个数字
              代表有多少个channel。因为我们只有灰度单色，所以是1，如果是彩色的RGB图片，这里应该是3。最后一个数字代表卷积核的数量，
              也就是这个卷积层会提取多少类的特征。
    :return:
    """
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    最大池化函数，使用2*2的最大池化，即将一个2*2的像素块将为1*1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即
    保留最显著的特征。因为希望整体上缩小图片尺寸，因此池化层的strides也设为横竖两个方向以2位步长。
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets(train_dir='/Users/yuxuecheng/Learn/MLData/MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型

    with tf.Session() as sess:
        sess.run(init)

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

