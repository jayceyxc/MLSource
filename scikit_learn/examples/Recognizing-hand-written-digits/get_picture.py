#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: get_picture.py.py
@time: 2017/4/24 上午10:00
"""

import urllib2

picture_url = "http://121.40.187.211:8080/index.php/Public/verify/0.7356425633784405"

if __name__ == '__main__':
    for index in range(1000):
        response = urllib2.urlopen(picture_url)
        content = response.read()
        picture_name = "data/%d.bmp" % index
        with open(picture_name, mode='w') as fd:
            fd.write(content)
