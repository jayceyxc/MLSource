#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: AdditiveGaussianNoiseAutoencoder.py
@time: 14/12/2017 08:56
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init( fan_in, fan_out, constant=1 ):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    """
    TensorFlow实现的去噪自编码器
    """
    def __init__( self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                  scale=0.1 ):
        """

        :param n_input: 输入变量数
        :param n_hidden: 隐藏层节点数
        :param transfer_function: 隐藏层激活函数，默认为softplus
        :param optimizer: 优化器，默认为Adam
        :param scale: 高斯噪声稀疏，默认为0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 下面定义网络结构
        # 创建一个维度为n_input的placeholder
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 建立一个能提取特征的隐藏层，将输入x加上噪声，即self.x + scale * tf.random_normal((n_input, ))，然后用tf.matmul将加了噪声的输入与隐藏层的权重
        # w1 相乘，并使用tf.add加上隐藏层的偏置b1，最后用self.transfer对结果进行激活函数处理
        self.hidden = self.transfer(
            tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        # 经过隐藏层后，在输出层进行数据复原、重建操作(即建立reconstruction层)，这里我们就不需要激活函数了，直接将隐藏层的输出self.hidden乘上输出的权重w2
        # 再加上输出层的偏置b2
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数，这里使用平方误差作为cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # 定义训练操作为优化器self.optimizer对损失self.cost进行优化
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights( self ):
        """
        参数初始化函数
        w1需要使用前面的xavier_init函数初始化，直接传入输入节点数和隐藏层节点数，然后xavier_init即可返回一个比较适合于softplus等
        激活函数的权重初始分布
        偏置b1只需要使用tf.zeros全部值为0即可
        :return:
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    def partial_fit( self, X ):
        """
        定义计算损失cost及执行一步训练的函数partial_fit。函数里只需要让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer
        输入的feed_dict包括输入数据X以及噪声的系数scale。

        该函数做的就是用一个batch数据进行训练并返回当前的损失cost
        :param X: 输入数据
        :return: 训练的损失cost
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})

        return cost

    def calc_total_cost(self, X):
        """
        执行一个计算图节点self.cost。
        这个函数在自编码器训练完毕后，在测试集上对模型性能进行评测时会用到
        :param X:
        :return:
        """
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform( self, X ):
        """
        该函数返回自编码器隐藏层的输出结果，目的是提供一个接口来获取抽象后的特征，自编码器隐藏层的最主要功能就是学习数据中的高阶特征
        :param X:
        :return:
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate( self, hidden=None ):
        """
        该函数将隐藏层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
        :param hidden:
        :return:
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])

        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct( self, X ):
        """
        该函数整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据，包括transform和generate两块
        :param X: 原始数据
        :return: 复原后的数据
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def get_weights( self ):
        """
        获取隐藏层的权重w1
        :return:
        """
        return self.sess.run(self.weights['w1'])

    def get_biases( self ):
        """
        获取隐藏层的偏置系数b1
        :return:
        """
        return self.sess.run(self.weights['b1'])


def standard_scale( X_train, X_test ):
    """
    对训练、测试数据进行标准化处理的函数。标准化即让数据变成0均值，且标准差为1的分布，方法就是先减去均值，再除以标准差
    注意：必须保证训练、测试数据都使用完全相同的Scaler，这样才能保证后面模型处理数据时的一致性，这也就是为什么现在训练
    数据上fit出一个共用的Scaler的原因

    :param X_train:
    :param X_test:
    :return:
    """
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test


def get_random_block_from_data( data, batch_size ):
    """
    随机获取block数据：取一个从0到len(data) - batch_size之间的随机整数，再以这个随机数作为block的起始位置，然后顺序渠道一个
    batch_size的数据。
    注意：这里属于不放回抽样，可以提高数据的利用效率
    :param data: 输入数据集
    :param batch_size: 需要获取的block的大小
    :return:
    """
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


if __name__ == '__main__':
    # 载入MNIST数据集
    mnist = input_data.read_data_sets(train_dir='/Users/yuxuecheng/Learn/MLData/MNIST_data', one_hot=True)
    # 使用之前定义的standard_scale函数对训练集、测试集进行标准化变换
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)  # 总训练样本数
    training_epochs = 20  # 最大训练的轮数设置为20
    batch_size = 128
    displat_step = 1  # 设置每隔一轮就显示一次损失cost

    # 创建一个AGN自编码器的示例，定义模型输入节点数n_input为784，自编码器的隐藏层节点数n_hidden为200，
    # 隐藏层的激活函数transfer_function为softplus，优化器optimizer为Adam并且学习速率为0.001，同时
    # 将噪声系数scale设为0.01
    auto_encoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = auto_encoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % displat_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(auto_encoder.calc_total_cost(X_test)))

    batch_xs = get_random_block_from_data(X_train, 5)
    print("batch_xs: ")
    print(batch_xs)
    hidden = auto_encoder.transform(batch_xs)
    print("hidden: ")
    print(hidden)
    generate = auto_encoder.generate(hidden)
    print("generate: ")
    print(generate)

