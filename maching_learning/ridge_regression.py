#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: ridge_regression.py

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

__author__ = 'yasaka'

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=1, solver='auto')
ridge_reg.fit(X, y)
print(ridge_reg.predict(1.5))

sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))





