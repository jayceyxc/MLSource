#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: classifier.py
@time: 2017/8/8 17:06
"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
from collections import Counter


# https://www.oschina.net/translate/big-picture-machine-learning
# https://mp.weixin.qq.com/s?__biz=MzA4MjEyNTA5Mw==&mid=2652565831&idx=1&sn=090c8e537877b52ed9c4cb58bfd876b4&chksm=8464d90db313501bb4707352c9aaeb0f22f31aa9f9f8a47482ba54fcec238a6470666a6b32ec&mpshare=1&scene=23&srcid=0808JaHzdkg6xmJcyuhOAT3C#rd


categories = ['comp.graphics', 'sci.space', 'rec.sport.baseball']
newsgroups_train = datasets.fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = datasets.fetch_20newsgroups(subset='test', categories=categories)
print('total texts in train:', len(newsgroups_train.data))
print('total texts in test:', len(newsgroups_test.data))

print('text', newsgroups_train.data[0])
print('category:', newsgroups_train.target[0])

vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

print("Total words:", len(vocab))

total_words = len(vocab)


# Now we have an index
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    return word2index

word2index = get_word_2_index(vocab)

print("Index of the word 'the':",word2index['the'])


def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i*batch_size+batch_size]
    categories = df.target[i * batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.

        results.append(y)

    return np.array(batches), np.array(results)


# Network Parameters
# 1st layer number of features
n_hidden_1 = 100
# 2nd layer number of features
n_hidden_2 = 100
# Words in vocab
n_input = total_words
# Categories: graphics, space and baseball
n_classes = 3

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1


def multilayer_perceptron(input_tensor, weights, biases):
    layer1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer1_addition = tf.add(layer1_multiplication, biases['b1'])
    layer1_activation = tf.nn.relu(layer1_addition)
    # Hidden layer with RELU activation
    layer2_multiplication = tf.matmul(layer1_activation, weights['h2'])
    layer2_addition = tf.add(layer2_multiplication, biases['b2'])
    layer2_activation = tf.nn.relu(layer2_addition)
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


input_tensor = tf.placeholder(tf.float32, [None, n_input], name='input')
output_tensor = tf.placeholder(tf.float32, [None, n_classes], name='output')

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)
# Define loss
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
loss = tf.reduce_mean(entropy_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # init the variables (normal distribution, remember?)
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: {0}, loss: {1}".format("%04d" % (epoch+1), "{:.9f}".format(avg_cost)))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
    print("Accuracy: ", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))