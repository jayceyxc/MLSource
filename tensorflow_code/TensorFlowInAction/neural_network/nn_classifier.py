#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: nn_classifier.py
@time: 2017/8/4 12:23
"""

"""
In this tutorial, we’ll use tf.contrib.learn to construct a neural network classifier and train it on the Iris data 
set to predict flower species based on sepal/petal geometry. You'll write code to perform the following five steps:

1. Load CSVs containing Iris training/test data into a TensorFlow Dataset
2. Construct a neural network classifier
3. Fit the model using the training data
4. Evaluate the accuracy of the model
5. Classify new samples
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # If the training and test sets aren't stored locally, download them
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "w") as f:
            f.write(raw)