#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: simple_regression.py
@time: 17/10/2017 13:31
"""

import tensorflow as tf
from tensorflow import rnn


class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        """
        #A Hyper-parameters 超参数
        #B Weight variables and input placeholders 权重变量和输入占位符
        #C Cost optimizer 成本优化器
        #D Auxiliary ops  辅助操作

        :param input_dim:
        :param seq_size:
        :param hidden_dim:
        """
        self.input_dim = input_dim # A
        self.seq_size = seq_size # A
        self.hidden_dim = hidden_dim # A

        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out') # B
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out') # B
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim]) # B
        self.y = tf.placeholder(tf.float32, [None, seq_size]) # B

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y)) # C
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost) # C

        self.saver = tf.train.Saver() # D

    def model( self ):
        """
        #A Create a LSTM cell
        创建一个LSTM单元
        #B Run the cell on the input to obtain tensors for outputs and states
        运行输入单元，获取输出和状态的张量
        #C Compute the output layer as a fully connected linear function
        将输出层计算为完全连接的线性函数

        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)  # A
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)  # B
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        # C
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train( self, train_x, train_y ):
        """
        #A Run the train op 1000 times
        训练1000次

        :param train_x:
        :param train_y:
        :return:
        """
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):  # A
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test( self, test_x ):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            print(output)


if __name__ == '__main__':
    """
    #A predicted result should be 1, 3, 5, 7
    #B predicted result should be 4, 9, 11, 13
    """
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    predictor.train(train_x, train_y)

    test_x = [[[1], [2], [3], [4]],  # A
              [[4], [5], [6], [7]]]  # B
    predictor.test(test_x)