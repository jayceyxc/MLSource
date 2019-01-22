#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: test.py
@time: 2017/8/8 16:47
"""

import tensorflow as tf

my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1, 3, 6])
    y = tf.constant([1, 1, 1])
    op = tf.add(x, y)
    result = sess.run(fetches=op)
    print(result)
