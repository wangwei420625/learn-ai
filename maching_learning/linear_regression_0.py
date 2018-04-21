#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_0.py
#解析解模型求解

import numpy as np
#画图的包
import matplotlib.pyplot as plt

__author__ = 'yasaka'

# 这里相当于是随机X维度X1，rand是随机均匀分布
X = 2 * np.random.rand(100, 1)
# 人为的设置真是的Y一列，np.random.randn(100, 1)是设置error，randn是标准正太分布
y = 4 + 3 * X + np.random.randn(100, 1)
# 整合X0和X1
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

# 常规等式求解theta
#np.linalg    线性代数相关的包
#np.linalg.inv    求逆
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# 创建测试集里面的X1
X_new = np.array([[0], [2]])
#整合X0  X1
X_new_b = np.c_[(np.ones((2, 1))), X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()



