#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_classification_tf_mlp.py
@time: 2017/8/9 10:04
"""

import tensorflow as tf
import numpy as np
from jieba import analyse
from sklearn.model_selection import train_test_split
import pickle

from utility import text_utility

vocabulary = dict()
all_categories = list()
total_words = 0
# with open("data" + os.sep + "category.txt", mode='r') as fd:
#     for line in fd:
#         line = line.strip()
#         # print(line)
#         all_categories.append(line)


# train_url_cat, train_url_content = text_utility.get_documents(current_path="data", pattern="train_*.xlsx")
# test_url_cat, test_url_content = text_utility.get_documents(current_path="data", pattern="test_*.xlsx")
#
# train_matrix = text_utility.my_extract(train_url_content.values(), vocabulary)
# test_matrix = text_utility.my_extract(test_url_content.values(), vocabulary)
#
# print('total texts in train:', len(train_url_content.values()))
# print('total texts in test:', len(test_url_content.values()))
#
# print('text: {0}'.format(train_url_content.values()[0]))
# print('category: {0}'.format(train_url_cat.values()[0]))


# Now we have an index
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    return word2index


# print("Index of the word u'公司':",word2index[u'公司'])


def get_batch(data, target, i, batch_size, word2index, topK=10):
    batches = []
    results = []
    texts = data[i * batch_size:i*batch_size+batch_size]
    categories = target[i * batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in analyse.extract_tags(sentence=text, topK=topK, withWeight=False):
            if word in word2index:
                layer[word2index[word]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((len(all_categories)), dtype=float)
        for i in range(0, len(all_categories), 1):
            if category == all_categories[i]:
                # print("{0} : {1}".format(i, category))
                y[i] = 1.

        results.append(y)

    return np.array(batches), np.array(results)


def multilayer_perceptron(input_tensor, weights, biases):
    # input_tensor * weights['h1']
    layer1_multiplication = tf.matmul(input_tensor, weights['h1'])
    # layer1_multiplication + biases['b1']
    layer1_addition = tf.add(layer1_multiplication, biases['b1'])
    layer1_activation = tf.nn.relu(layer1_addition)
    layer1_drop = tf.nn.dropout(layer1_activation, keep_prob=0.8)
    # Hidden layer with RELU activation
    layer2_multiplication = tf.matmul(layer1_drop, weights['h2'])
    layer2_addition = tf.add(layer2_multiplication, biases['b2'])
    layer2_activation = tf.nn.relu(layer2_addition)
    layer2_drop = tf.nn.dropout(layer2_activation, keep_prob=0.8)
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer2_drop, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


def train_and_test(train_data, train_cat, test_data, test_cat, word2index):
    """

    :param train_data: the train data, type list
    :param train_cat: the corresponding categories of train data, type list
    :param test_data: the test data, type list
    :param test_cat: the corresponding categories of test data, type list
    :return:
    """

    global all_categories
    global total_words

    # Network Parameters
    # 1st layer number of features
    n_hidden_1 = 100
    # 2nd layer number of features
    n_hidden_2 = 100
    # Words in vocab
    n_input = total_words
    # Categories: graphics, space and baseball
    n_classes = len(all_categories)

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 20
    display_step = 1

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
    cat = tf.nn.softmax(prediction)
    tf.add_to_collection('pred_network', cat)
    # Define loss
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
    loss = tf.reduce_mean(entropy_loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型

    # Launch the graph
    with tf.Session() as sess:
        # init the variables (normal distribution, remember?)
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(train_data) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = get_batch(train_data, train_cat, i, batch_size, word2index)
                # Run optimization op (backprop) and cost op (to get loss value)
                c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor:batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: {0}, loss: {1}".format("%04d" % (epoch+1), "{:.9f}".format(avg_cost)))

        print("Optimization Finished!")
        # 将模型保存到model/url_classfication.ckpt文件
        saver_path = saver.save(sess, "model/url_classfication_test.ckpt")
        print("Model saved in file:", saver_path)

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_test_data = len(test_data)
        batch_x_test, batch_y_test = get_batch(test_data, test_cat, 0, total_test_data, word2index)
        print("Accuracy: ", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))


def train_model():
    global all_categories
    global total_words
    url_cat, url_content = text_utility.get_documents(current_path="../tensorflow/data", pattern="Query_train_content.xlsx")
    all_categories = list(set(url_cat.values()))
    text_utility.my_extract(url_content.values(), vocabulary)

    url_cat = list(url_cat.values())
    url_content = list(url_content.values())
    cat_train, cat_test, content_train, content_test = train_test_split(url_cat, url_content, test_size=0.2)
    print('total texts in train:', len(content_train))
    print('total texts in test:', len(content_test))
    print('total categories: ', len(all_categories))
    print('test categories: ', len(set(cat_test)))

    print("Total words:", len(vocabulary))

    total_words = len(vocabulary)

    with open("vocabulary.txt", mode='wb') as fd:
        pickle.dump(vocabulary, fd)
        fd.flush()

    with open("categories.txt", mode='wb') as fd:
        pickle.dump(all_categories, fd)
        fd.flush()

    word2index = get_word_2_index(vocabulary)

    train_and_test(url_content, url_cat, content_test, cat_test, word2index)


def predict(input_content, results):
    for content in input_content:
        print(np.nonzero(content))
    # print(results)

    all_categories = []
    with open("categories.txt", mode='rb') as fd:
        all_categories = pickle.load(fd)

    print(all_categories)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model/url_classfication_test.ckpt.meta')
        new_saver.restore(sess, 'model/url_classfication_test.ckpt')
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('input').outputs[0]
        input_y = graph.get_operation_by_name('output').outputs[0]
        cat = tf.get_collection('pred_network')[0]

        predict_cat = sess.run(tf.argmax(cat, axis=1), feed_dict={input_x: input_content, input_y: results})
        print(predict_cat)
        for index in predict_cat:
            print(all_categories[index])

        # print(results)
        # print(cat)


def predict_for_file(file_name, topK=10):
    vocabulary = {}
    with open("vocabulary.txt", mode='rb') as fd:
        vocabulary = pickle.load(fd)

    # for key, value in vocabulary.items():
    #     print("%d:%s" % (value, key))

    total_words = len(vocabulary)
    word2index = get_word_2_index(vocabulary)
    batches = []
    results = []
    with open(file_name, mode='r') as fd:
        for line in fd:
            line = line.strip()
            layer = np.zeros(total_words, dtype=float)
            for word in analyse.extract_tags(sentence=line, topK=topK, withWeight=False):
                if word in word2index:
                    layer[word2index[word]] += 1

            batches.append(layer)
            y = np.zeros((152), dtype=float)
            results.append(y)

    predict(np.array(batches), np.array(results))


if __name__ == '__main__':
    train_model()
    # predict_for_file("test.txt")
