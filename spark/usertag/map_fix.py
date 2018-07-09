#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: map_fix.py
@time: 2017/6/2 上午11:50
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('.')

import utility
from parse_info_mdf import parse_cookie, parse_url, trans_str, get_field_indexes
from rule_manager import RuleParser
import config


field_list = ['NUM', 'ADSL', 'HOST', 'URL', 'REFER', 'UA', 'COOKIE', 'TIME', 'SourceIP', 'DestIP']

NUM, ADSL, HOST, URL, REFER, UA, COOKIE, TIMESTAMP, SourceIP, DEST_IP = get_field_indexes('map_fix', field_list)

rule_parser = RuleParser(net_type='fix')
label_rules = rule_parser.rules

exclude_char=set("*,<&%$#@~!^()<>*{}\"\'[]=+|\\")


def update_label_id(url, label_id):
    """
    根据url更新label_id这个集合，将url所属的标签更新到label_id集合中
    :param url: 用户访问的url，根据该url获取标签
    :param label_id: 用户的标签集合
    :return: 
    """
    url = trans_str(url)
    host = utility.url_to_host(url)
    domain = utility.host_to_domain(host)
    try:
        if url in label_rules['url']:
            label_id.update(label_rules['url'][url])
    except KeyError:
        pass

    try:
        if host in label_rules['host']:
            label_id.update(label_rules['host'][host])
    except KeyError:
        pass

    try:
        if domain in label_rules['domain']:
            label_id.update(label_rules['domain'][domain])
    except KeyError:
        pass

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    try:
        _, line = line.split('\t', 1)
    except ValueError:
        continue
    segs = [s.strip() for s in line.split('\05')]
    if len(segs) < NUM or not segs[ADSL]:
        continue

    info.adsl = segs[ADSL]
    # DOMAIN 字段实际为host
    info.host = segs[HOST]
    info.raw_url = segs[URL]
    # 对URL进行urllib2.decode解码，将其中的%转码过的信息还原为原始字符串
    info.url = trans_str(info.raw_url)

    info.url = utility.spider_url_to_dpi_url(info.url)
    info.domain = utility.host_to_domain(info.host)

    if config.is_debug():
        print u"test" + u",".join([info.adsl, info.url, info.host, info.domain])

    label_id = set()
    update_label_id(info.url, label_id)
    update_label_id(info.refer, label_id)

    if len(label_id) > 0:
        try:
            print u"\t".join([info.adsl, u",".join(label_id)])
        except UnicodeEncodeError:
            pass
