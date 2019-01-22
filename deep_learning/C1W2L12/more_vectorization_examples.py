#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: more_vectorization_examples.py
@time: 2018/9/21 8:48 PM
"""

import numpy as np
import math
import time

n = 1000000
v = np.random.rand(n)
u = np.zeros(n)
print(v.shape)
print(u.shape)
print(type(v))
print(type(u))
tic = time.time()
for i in range(n):
    u[i] = math.exp(v[i])
toc = time.time()
print("For loop: " + str(1000 * (toc - tic)) + "ms")

print(v.shape)
print(u.shape)

tic = time.time()
u = np.exp(v)
toc = time.time()
print("numpy loop: " + str(1000 * (toc - tic)) + "ms")

print(v.shape)
print(u.shape)
