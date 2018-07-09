#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: rule_manager.py
@time: 2017/6/3 下午11:31
"""



""" 识别规则文件（rule***.csv）解析模块
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('.')
import re
import glob
import csv
from collections import OrderedDict


class RuleParser(object):
    """ 识别规则解析器类型 """

    def __init__(self, net_type='fix'):
        """ 规则对象初始化时设置各成员变量初始值
        :param net_type: 规则针对的网络类型，fix-固网，mobile-移动网
        """
        # 网络类型
        self.net_type = net_type
        # 去重后的原始规则
        self.rules = OrderedDict()
        # 解析后的规则，分4种类型
        self.parsed_rules = {'domain': [], 'url': []}
        # 规则文件头（csv第一行）
        self.csv_head = []
        # 去重过程中使用的key各字段的分隔符
        self.key_sep = '~'
        # 可以定义的关键字类型，当前可以使用顶级域名、二级域名、具体的url这三种。
        self.keys = set(['domain', 'host', 'url'])
        # 对象初始化最后生成当前各种规则（原始rules及解析好的parsed_rules）
        self.generate_current()

    def parse_rules(self, path):
        with open(path) as f:
            csv_reader = csv.reader(f)
            for index, segs in enumerate(csv_reader):
                if not index:
                    self.csv_head = segs
                    continue
                segs = [s.strip() for s in segs]
                if segs[0] != '1' or segs[1] not in ('all', self.net_type):
                    continue
                label_id = segs[2]
                rule_types = segs[3].split()
                url_patterns = segs[4].split(",")
                for rule_type in rule_types:
                    if rule_type not in self.keys:
                        continue
                    if rule_type in self.rules:
                        for url_pattern in url_patterns:
                            if url_pattern in self.rules[rule_type]:
                                self.rules[rule_type][url_pattern].add(label_id)
                            else:
                                self.rules[rule_type][url_pattern] = set()
                                self.rules[rule_type][url_pattern].add(label_id)
                    else:
                        self.rules[rule_type] = dict()
                        for url_pattern in url_patterns:
                            if url_pattern in self.rules[rule_type]:
                                self.rules[rule_type][url_pattern].add(label_id)
                            else:
                                self.rules[rule_type][url_pattern] = set()
                                self.rules[rule_type][url_pattern].add(label_id)


    def generate_current(self):

        for filename in glob.glob('rule*.csv'):
            # print filename
            with open(filename) as f:
                csv_reader = csv.reader(f)
                for index, segs in enumerate(csv_reader):
                    if not index:
                        self.csv_head = segs
                        continue
                    segs = [s.strip() for s in segs]
                    if segs[0] != '1' or segs[1] not in ('all', self.net_type):
                        continue
                    label_id = segs[2]
                    rule_types = segs[3].split()
                    url_patterns = segs[4].split(",")
                    for rule_type in rule_types:
                        if rule_type not in self.keys:
                            continue
                        if rule_type in self.rules:
                            for url_pattern in url_patterns:
                                if url_pattern in self.rules[rule_type]:
                                    self.rules[rule_type][url_pattern].add(label_id)
                                else:
                                    self.rules[rule_type][url_pattern] = set()
                                    self.rules[rule_type][url_pattern].add(label_id)
                        else:
                            self.rules[rule_type] = dict()
                            for url_pattern in url_patterns:
                                if url_pattern in self.rules[rule_type]:
                                    self.rules[rule_type][url_pattern].add(label_id)
                                else:
                                    self.rules[rule_type][url_pattern] = set()
                                    self.rules[rule_type][url_pattern].add(label_id)


    def parse_rules(self):
        # if filename:
        #     with open(filename, 'w') as cf:
        #         current_file = csv.writer(cf)
        #         current_file.writerow(self.csv_head)
        rules = self.rules
        for account_type in rules:
            for rule_type in rules[account_type]:
                value_clct = []
                for pattern_key in rules[account_type][rule_type]:
                    url_pattern, para_key = pattern_key.split(self.key_sep)
                    if rule_type != 'domain':
                        url_pattern = re.compile(url_pattern)

                    value_pattern, group_index = rules[account_type][rule_type][pattern_key]
                    if account_type.startswith('sdk'):
                        value_pattern = [re.compile(p_str) for p_str in value_pattern.split()]
                    else:
                        value_pattern = re.compile(value_pattern)

                    value = (url_pattern, para_key, value_pattern, group_index)
                    value_clct.append(value)

                self.parsed_rules[rule_type].append((account_type, value_clct))
                    # if filename:
                    #     segs = ['1', self.net_type, account_type, rule_type] + url_keys
                    #     segs.extend(value_ptns)
                    #     current_file.writerow(segs)


if __name__ == '__main__':
    # 展示各条生效规则到标准输出
    for n_type in 'fix', 'mobile':
        # print '#' * 20, n_type, '#' * 20
        rule_parser = RuleParser(n_type)
        # print rule_parser.rules
        print '=' * 20, n_type, 'rules', '=' * 20
        for k, v in rule_parser.rules.iteritems():
            print n_type, k
            print v
            print '-' * 20
