#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: stochastic_gradient_descent.py
#随机梯度下降

import numpy as np

__author__ = 'yasaka'

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

n_epochs = 500
t0, t1 = 5, 50  # 超参数

m = 100


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)


for epoch in range(n_epochs):
    for i in range(m):
        #生成100以内的一个随机数
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        #梯度   一条样本对应多个维度的梯度
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        #学习率
        learning_rate = learning_schedule(epoch*m + i)
        theta = theta - learning_rate * gradients

print(theta)






