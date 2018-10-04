#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: mnist_softmax.py
@time: 2018/9/8 10:42 PM
"""

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def print_mnist_data(mnist_data):
    """
    查看mnist数据集的大小
    :param mnist_data:
    :return:
    """
    # 查看训练数据的大小
    print(mnist_data.train.images.shape)
    print(mnist_data.train.labels.shape)

    # 查看验证数据集的大小
    print(mnist_data.validation.images.shape)
    print(mnist_data.validation.labels.shape)

    # 查看测试数据的大小
    print(mnist_data.test.images.shape)
    print(mnist_data.test.labels.shape)


def save_as_pics(mnist):
    # 保存mnist数据集的图片
    # 打印第0张图片的向量表示
    print(mnist.train.images[0, :])
    print(mnist.train.labels[0, :])
    save_dir = '/Users/yuxuecheng/TF_data/MNIST_data/pictures/'
    # 将图片保存在/Users/yuxuecheng/TF_data/MNIST_data/pictures目录下，如果目录不存在，会自动创建
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 保存前20张图片
    for i in range(20):
        # 请注意, mnist.train.images[i,:]就表示第i张图片（序号从0开始）
        image_array = mnist.train.images[i, :]
        # TensorFlow中MNIST图片是一个784维的向量，我们重新把它还原为28*28维的图像
        image_array = image_array.reshape(28, 28)
        # 保存文件的格式为
        # mnist_train_0.jpg,mnist_train_1.jpg, ... ,mnist_train_19.jpg,
        filename = save_dir + 'mnist_train_%d.jpg' % i
        # 将image_array保存为图片
        # 先用scipy.misc.toimage转换为图像，再调用save直接保存
        scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

        # 得到one-hot表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        one_hot_label = mnist.train.labels[i, :]
        #
        label = np.argmax(one_hot_label)
        print('mnist_train_%d.jpg, label: %d' % (i, label))


def softmax_regression(mnist):
    # 创建x，x是一个占位符（placeholder），代表待识别的图片
    x = tf.placeholder(tf.float32, [None, 784])

    # W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
    # 在TensorFlow中，变量的参数用tf.V
    W = tf.Variable(tf.zeros([784, 10]))
    # b是有一个Softmax模型的参数，一般叫做"偏置项"
    b = tf.Variable(tf.zeros([10]))

    # y表示模型的输出
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # y_是实际的图像标签，同样以占位符表示
    y_ = tf.placeholder(tf.float32, [None, 10])

    # y是模型的输出，y_是实际的图像标签，注意y_是one-hot表示的，下面会根据y和y_构造交叉熵损失
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

    # 有了损失，就可以用梯度下降方法针对模型的参数（W和b）进行优化
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 创建一个Session。只有在Session中才能运行优化步骤train_step
    sess = tf.InteractiveSession()
    # 运行之前必须要初始化所有变量，分配内存
    tf.global_variables_initializer().run()
    # 进行1000步梯度下降
    for _ in range(1000):
        # 在mnist.train中取100个训练数据
        # batch_xs是形状为(100,784)的图像数据，batch_ys是形如(100,10)的实际标签
        # batch_xs和batch_ys对应着两个占位符x和y_
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 在Session中运行train_step，运行时要传入占位符的值
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 正确的预测结果
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 计算预测准确率，它们都是Tensor
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 在Session中运行Tensor可以得到Tensor的值
    # 这里是获取最终模型的准确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    # cong /Users/yuxuecheng/TF_data/MNIST_data/中读取MNIST数据。这条语句在数据不存在时，会自动执行下载
    mnist = input_data.read_data_sets("/Users/yuxuecheng/TF_data/MNIST_data/", one_hot=True)
    print_mnist_data(mnist)
    softmax_regression(mnist)

