#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: tf_computation.py
@time: 15/11/2017 09:10
"""

import tensorflow as tf
import random


# Basic computations
def basic_compute():
    a = tf.Variable(3)
    b = tf.Variable(4)

    c = tf.multiply(a, b)
    d = tf.add(a, c)
    print (c)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # this is important

        c_value = sess.run(c)
        d_value = sess.run(d)

        print (c_value, d_value)


# Save Model
def save_model():
    a = tf.Variable(5)
    b = tf.Variable(4, name="my_variable")

    # set the value of a to 3
    op = tf.assign(a, 3)

    # create saver object
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(op)

        print ("a:", sess.run(a))
        print ("my_variable:", sess.run(b))

        # use saver object to save variables
        # within the context of the current session
        saver.save(sess, "/tmp/my_model.ckpt")


def load_module():
    # Load module
    # make a dummy variable
    # the value is arbitrary, here just zero
    # but the shape must the the same as in the saved model
    a = tf.Variable(0)
    c = tf.Variable(0, name="my_variable")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # use saver object to load variables from the saved model
        saver.restore(sess, "/tmp/my_model.ckpt")

        print ("a:", sess.run(a))
        print ("my_variable:", sess.run(c))


def visualize():
    # tensorboard --logdir=/tmp/summary
    a = tf.Variable(5, name="a")
    b = tf.Variable(10, name="b")

    c = tf.multiply(a, b, name="result")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print (sess.run(c))

        fw = tf.summary.FileWriter("/tmp/summary", sess.graph)


def namespace_scope():
    with tf.name_scope('primitives') as scope:
        a = tf.Variable(5, name='a')
        b = tf.Variable(10, name='b')

    with tf.name_scope('fancy_pants_procedure') as scope:
        # this procedure has no significant interpretation
        # and was purely made to illustrate why you might want
        # to work at a higher level of abstraction
        c = tf.multiply(a, b)

        with tf.name_scope('very_mean_reduction') as scope:
            d = tf.reduce_mean([a, b, c])

        e = tf.add(c, d)

    with tf.name_scope('not_so_fancy_procedure') as scope:
        # this procedure suffers from imposter syndrome
        d = tf.add(a, b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print (sess.run(c))
        print (sess.run(e))

        fw = tf.summary.FileWriter("/tmp/summary2", sess.graph)


def visualize_changed_variable():
    a = tf.Variable(5, name="a")
    b = tf.Variable(10, name="b")

    # set the intial value of c to be the product of a and b
    # in order to write a summary of c, c must be a variable
    init_value = tf.multiply(a, b, name="result")
    c = tf.Variable(init_value, name="ChangingNumber")

    # update the value of c by incrementing it by a placeholder number
    number = tf.placeholder(tf.int32, shape=[], name="number")
    c_update = tf.assign(c, tf.add(c, number))

    # create a summary to track to progress of c
    tf.summary.scalar("ChangingNumber", c)

    # in case we want to track multiple summaries
    # merge all summaries into a single operation
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # initialize our summary file writer
        fw = tf.summary.FileWriter("/tmp/summary3", sess.graph)

        # do 'training' operation
        for step in range(1000):
            # set placeholder number somewhere between 0 and 100
            num = int(random.random() * 100)
            sess.run(c_update, feed_dict={number: num})

            # compute summary
            summary = sess.run(summary_op)

            # add merged summaries to filewriter,
            # so they are saved to disk
            fw.add_summary(summary, step)


if __name__ == '__main__':
    # load_module()
    # visualize()
    # namespace_scope()
    visualize_changed_variable()