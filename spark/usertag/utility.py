#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: my_thread.py
@time: 2017/6/2 上午11:50
"""

""" 各种操作字符串等的简易工具函数集
"""


import time
import types
import hashlib

arrow_mark = '<=='


def spider_url_to_dpi_url(url):
    """ 去掉URL可能存在的http、https头部
    :param url: 需处理的URL
    :return: 处理好后的URL
    """
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    return url


def dpi_url_to_spider_url(url):
    """ 恢复被去掉http、https头部的URL
    :param url: 需处理的URL
    :return: 处理好后的URL
    """
    if not (url.startswith('http://') or url.startswith('https://')):
        url = url.lstrip(':/')
        url = 'http://' + url
    if url.count('/') == 2:
        url += '/'
    return url


def url_to_host(url):
    """ 提取URL中的host
    :param url: 需提取host的URL
    :return: 提取出的host
    """
    url = spider_url_to_dpi_url(url)
    url_host = url.split('/')[0]
    return url_host


# 各种顶级域名
__suffix__ = ['com', 'cn', 'net', 'org', 'edu', 'vc', 'biz',
              'in', 'co', 'top', 'tech', 'club', 'tv', 'to']
__suffix__ = set(__suffix__)


def host_to_domain(host):
    """ 提取host中的domain
    :param host: 需提取domain的host
    :return: 提取出的domain
    """
    try:
        host, port = host.split(':')
    except ValueError:
        host, port = host, None
    segs = host.split('.')
    if len(segs) == 4 and ''.join(segs).isdigit():
        return host
    else:
        domain_tokens = []
        for token in segs[::-1]:
            domain_tokens.append(token)
            if token not in __suffix__:
                break
        return '.'.join(domain_tokens[::-1])


def unix_time_to_str(value, format_str='%Y%m%d %H:%M:%S'):
    """ 将Unix时间（数值）转化为时间字符串
    :param value: 需转化的Unix时间数值（int/float）
    :param format_str: 转化成的时间字符串的格式
    :return: 转化成的时间字符串
    """
    if type(value) is types.IntType:
        value = float(value)
    format = format_str
    value = time.localtime(value)
    return time.strftime(format, value)


def str_to_unix_time(dt, format_str='%Y%m%d %H:%M:%S'):
    """ 将时间字符串转化为Unix时间
    :param dt: 需转化的时间字符串
    :param format_str: 需转化的时间字符串的格式
    :return: 转化成的Unix时间（int，单位：秒）
    """
    time.strptime(dt, format_str)
    s = time.mktime(time.strptime(dt, format_str))
    return int(s)


def md5_value(key):
    """ 将某字符串进行MD5加密
    :param key: 需加密的字符串
    :return: MD5加密后的字符串
    """
    md5value = hashlib.md5()
    md5value.update(key)
    return md5value.hexdigest()


def count_one(counter_dict, key):
    """ 对某字典型计数器容器进行加一操作
    :param counter_dict: 目标计数器容器（dict型）
    :param key: 需加一的计数器key
    :return: 无
    """
    # if not key:
    #     return
    try:
        counter_dict[key] += 1
    except KeyError:
        counter_dict[key] = 1


if __name__ == '__main__':
    print(host_to_domain("games.sina.com.cn"))