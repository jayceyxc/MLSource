#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 本模块包含各种工具性解析函数，解析COOKIE、URL中的键值对；对其包含的信息进行转码；
对field.ini配置文件进行解析；对黑白ID名单进行解析；对present列表文件进行解析等
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('.')
import urllib
import csv
from collections import OrderedDict
try:
    # 自定义模块放在try-catch中是为在只使用部分不涉及该模块的函数时，
    # 避免仍需import该模块，比如本模块只有get_field_indexes()涉及config_parser
    from config import config_parser
except ImportError:
    pass


def parse_cookie(cookie):
    """ 将COOKIE转为dict保存（COOKIE默认以';'分隔各键值对，以'='分隔键和值）
    :param cookie: 需提取键值对的COOKIE
    :return: 提取出的键值对（均为字符串）组成的dict，若提取不成功则返回空dict
    """
    try:
        segs = cookie.split(';')
        cookie_dict = {}
        for i in segs:
            key_, value_ = i.split('=', 1)
            if len(value_) < 4:
                continue
            cookie_dict[key_.strip()] = value_.strip()
        return cookie_dict
    except:
        return {}


def trans_str(raw):
    """ 对URL、COOKIE、UA中的特殊字符进行转码（包括中文），最终转为utf-8
    :param raw: 需转码的原始字符串
    :return: 转码后的字符串（若转码出错，则返回空字符串）
    """
    try:
        if raw.find('%') >= 0:
            raw = urllib.unquote(raw)
        if raw.find('%') >= 0:
            raw = urllib.unquote(raw)
        # 此处是为解析%uABCD类型的非标准url编码，该种编码是直接将Unicode编码点ABCD以%u标识
        if raw.find('%u') >= 0:
            raw = raw.replace('%u', '\\u')
        if raw.find('\\u') >= 0:
            raw = raw.decode('unicode_escape').encode('utf-8')
        # raw = raw.translate(None, '\n\r\t')
        return raw
    except Exception:
        return ''


def parse_url(url):
    """ 将URL的'?'后的query字符串键值对转为dict保存（query默认以'&'分隔各键值对，以'='分隔键和值）
    :param url: 需提取键值对的URL
    :return: 提取出的键值对（均为字符串）组成的dict，若提取不成功则返回可解析部分dict
    
    >>> print parse_url('order.jd.com/center/list.acti��=1&s=4096&page=9363')
    {'s': '4096', 'page': '9363'}
    >>> print parse_url('order.jd.com/center/list.acti?=1&s=4096&page=9363')
    {'s': '4096', 'page': '9363'}
    """
    url_dict = {}
    try:
        segs = url.split('?')[1]
    except IndexError:
        try:
            segs = url.split('��')[1]
        except IndexError:
            return url_dict

    segs = segs.split('&')
    for i in segs:
        try:
            key_, value_ = i.split('=', 1)
        except ValueError:
            continue
        # 用户账号长度不会小于4
        if len(value_) < 4 or not key_:
            continue
        url_dict[key_.strip()] = value_.strip()

    return url_dict


if __name__ == "__main__":
    pass
