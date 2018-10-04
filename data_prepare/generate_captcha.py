#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: generate_captcha.py
@time: 14/11/2017 16:48
"""

import os
from random import Random
import time
from captcha.image import ImageCaptcha
from random_words import RandomWords

if __name__ == '__main__':
    image = ImageCaptcha()
    rw = RandomWords()
    random = Random()
    random.seed(time.time())
    valid_charsets = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')

    # for i in range(0, 10000):
    #     word = rw.random_word()
    #     # data = image.generate(word)
    #     image.write(word, os.path.join(os.getcwd(), "captcha", word + ".png"))
    #
    # for i in range(0, 2000):
    #     number = str(random.randint(1000, 9999))
    #     image.write(number, os.path.join(os.getcwd(), "captcha", number + ".png"))
    for i in range(0, 10000):
        word = ""
        for i in range(6):
            index = random.randint(0, len(valid_charsets) - 1)
            word = word + valid_charsets[index]

        print word
        image.write(word, os.path.join(os.getcwd(), "mix_captcha", word + ".png"))




