#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: map_mobile.py
@time: 2017/6/2 上午11:50
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('.')

import utility
import interface
from parse_info_mdf import parse_url, trans_str, get_field_indexes
from rule_manager import RuleParser
import config


class DpiConfig(object):
    def __init__(self, source_str):
        field_list = ['NUM', 'IMSI', 'MDN', 'MEID', 'BSID', 'STARTTIME', 'ENDTIME', 'UA', 'URL', 'HOST', 'REFER', 'SourceIP']
        self.NUM, self.IMSI, self.MDN, self.MEID, self.BSID, self.STARTTIME, self.ENDTIME, \
        self.UA, self.URL, self.HOST, self.REFER, self.SOURCEIP = get_field_indexes(source_str, field_list)

# 移动网Mapper分析模块可以同事处理3G和4G两种数据。
dpi_3g = DpiConfig('map_mobile')
dpi_4g = DpiConfig('map_mobile_4G')

info = interface.Interface('mobile')

rule_parser = RuleParser(net_type='mobile')
label_rules = rule_parser.rules

if config.is_debug():
    try:
        print label_rules
    except KeyError:
        pass

    try:
        print label_rules['url']
    except KeyError:
        pass

    try:
        print label_rules['host']
    except KeyError:
        pass

    try:
        print label_rules['domain']
    except KeyError:
        pass

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
    segs = [s.strip() for s in line.split('|')]
    # 通过长度判断该数据是属于3G还是4G
    seg_length = len(segs)
    if seg_length == dpi_3g.NUM:
        data_source = dpi_3g
    elif seg_length == dpi_4g.NUM:
        data_source = dpi_4g
    # 无法确认，说明情况异常，不处理直接退出
    else:
        continue

    info.imsi = segs[data_source.IMSI]
    info.mdn = segs[data_source.MDN]
    info.meid = segs[data_source.MEID]
    info.raw_url = segs[data_source.URL]
    info.url = trans_str(info.raw_url)

    # 必须同时存在以下字段才可以继续：IMSI MDN MEID URL
    if not info.mdn:
        continue

    if not (info.imsi and info.meid and info.url):
        continue

    info.bsid = segs[data_source.BSID]
    info.host = segs[data_source.HOST]
    info.refer = segs[data_source.REFER]

    info.url = utility.spider_url_to_dpi_url(info.url)
    info.domain = utility.host_to_domain(info.host)


    if config.is_debug():
        print u"test" + u",".join([info.mdn, info.url, info.host, info.domain])

    label_id = set()
    update_label_id(info.url, label_id)
    update_label_id(info.refer, label_id)

    if len(label_id) > 0:
        try:
            print u"\t".join([info.mdn, u",".join(label_id)])
        except UnicodeEncodeError:
            pass

