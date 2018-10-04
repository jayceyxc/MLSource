#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: broadcasting_example.py
@time: 2018/9/21 9:25 PM
"""

import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
print(A)
print(A.shape)

B = np.sum(A, axis=0)
print(B)

C = 100 * A / B
print(C)
