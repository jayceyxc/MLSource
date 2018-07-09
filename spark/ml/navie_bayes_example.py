#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: navie_bayes_example.py
@time: 2017/7/19 17:21
"""

from pyspark import SparkContext, SparkConf

# Loading training data
conf = SparkConf().setAppName("Spark Test").setMaster("spark://192.168.3.110:7077")
sc = SparkContext(conf=conf)

lines = sc.textFile("data.txt")
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)
print totalLength