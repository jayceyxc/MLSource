#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_classification_fasttext.py
@time: 2017/8/7 12:18
"""

import os
import sys

import fasttext
import jieba
from jieba import analyse

from utility import text_utility


def format_data(url_cat, url_content, out=sys.stdout):
    for url in url_content.keys():
        try:
            cat = url_cat[url]
            content = url_content[url]
            word_vec = jieba.cut(sentence=content, cut_all=False)
            outline = " ".join(word_vec)
            outline = "__label__" + cat + "\t" + outline + os.linesep
            out.write(outline)
        except KeyError as ke:
            pass


def url_classification_fasttext():
    url_cat, url_content = text_utility.get_documents(current_path="data", pattern="train*.xlsx")
    with open("fasttext_train_nocutall.txt", mode="w") as fd:
        format_data(url_cat, url_content, fd)
        fd.flush()

    classifier = fasttext.supervised(input_file="fasttext_train_nocutall.txt", output="fasttext_nocutall.model", label_prefix="__label__")
    # classifier = fasttext.load_model("fasttext.model.bin", label_prefix="__label__")
    print(len(classifier.labels))
    for class_name in classifier.labels:
        print(class_name)

    texts = list()
    with open("test.txt", mode="r") as fd:
        for line in fd:
            line = line.strip()
            segs = line.split(',')
            if len(segs) != 6:
                continue
            url, title, keywords, desc, a_content, p_content = line.split(',')
            content = " ".join([title, keywords, desc, a_content, p_content])
            word_vec = [word for word in jieba.cut(content, cut_all=False)]
            if len(word_vec) == 0:
                continue
            test_content = " ".join(word_vec)
            print(url, test_content)
            texts.append(test_content)
            # predict函数的输入需要使用list类型

    label_list = classifier.predict_proba(texts, len(classifier.labels))
    for label in label_list:
        for value in label:
            print(value[0], value[1])


if __name__ == "__main__":
    jieba.load_userdict("dict" + os.sep + "user.dict")
    analyse.set_stop_words("dict" + os.sep +"stop_words.txt")
    url_classification_fasttext()


