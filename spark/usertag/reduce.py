#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: reduce.py.py
@time: 2017/6/2 上午11:50
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('.')
import config

cur_account = None
label_count = dict()


def print_label():
    if cur_account is not None and len(label_count) > 0:
        sorted_label_list = sorted(label_count.items(), key=lambda d: d[1], reverse=True)
        print "%s:%s" % (cur_account, sorted_label_list[:config.get_max_label_number()])

for line in sys.stdin:
    line = line.strip()
    account, labels = line.split("\t", 1)
    label_list = labels.split(",")

    if cur_account == account:
        for label in label_list:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
    else:
        try:
            if cur_account:
                for key in label_count.keys():
                    if label_count[key] < config.get_threshold():
                        del label_count[key]
                if len(label_count) == 0:
                    continue
                print_label()

            label_count.clear()
            cur_account = account
            for label in label_list:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
        except:
            continue