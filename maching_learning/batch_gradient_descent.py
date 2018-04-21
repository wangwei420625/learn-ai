#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: batch_gradient_descent.py
#梯度下降

import numpy as np

__author__ = 'yasaka'



# 产生100行1列均匀分布随机数组
X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)


#np.ones(100,1) #创建指定长度的(100行1列二维)的全1数组
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)

learning_rate = 0.1
n_iterations = 1000
m = 100

#标准的正态分布
theta = np.random.randn(2, 1)
count = 0

for iteration in range(n_iterations):
    count += 1
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - learning_rate * gradients

print(count)
print(theta)









