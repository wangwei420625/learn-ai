#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_1.py
#梯度下降

import numpy as np
from sklearn.linear_model import LinearRegression

__author__ = 'ww'

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))
