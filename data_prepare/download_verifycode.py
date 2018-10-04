#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: download_verifycode.py
@time: 14/11/2017 16:35
"""

import requests
from bs4 import BeautifulSoup
import urllib2
import os
from random import Random
import time


headers = {
    "Accept": "text/html, application/xhtml+xml, */*",
    "Accept-Language": "zh-CN",
    "User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.89 Safari/537.36',
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.zhihu.com",
    "DNT": "1",
    "Connection": "Keep-Alive"
}


if __name__ == '__main__':
    url = "http://reg.email.163.com/unireg/call.do?cmd=register.verifyCode&v=common/verifycode/vc_en&vt=main_acode"
    session = requests.session()
    save_path = os.path.join(os.getcwd(), "verifycode")
    rand = Random()
    rand.seed(time.time())
    for i in range(0, 10000, 1):
        captcha_content = session.get(url).content
        filename = "captcha_%d.jpg" % i
        full_filename = os.path.join(save_path, filename)
        with open(full_filename, mode="wb") as fd:
            fd.write(captcha_content)
        time.sleep(rand.random())