#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: vectorization_demo.py
@time: 2018/9/21 8:38 PM
"""

import numpy as np
import time

a = np.array([1, 2, 3, 4])
print(a)

dimension = 1000000
a = np.random.rand(dimension)
b = np.random.rand(dimension)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version: " + str(1000 * (toc - tic)) + "ms")

c = 0
tic = time.time()
for i in range(dimension):
    c += a[i] * b[i]

toc = time.time()
print(c)
print("For loop: " + str(1000 * (toc - tic)) + "ms")
