#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: softmax_number_classify.py
@time: 2018/7/6 9:39 AM
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/Users/yuxuecheng/Learn/MLData/MNIST_data", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

with tf.Session() as sess:
    # sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # print(batch_xs.shape)
        # print(batch_ys.shape)
        train_step.run({x: batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
