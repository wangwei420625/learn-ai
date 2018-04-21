#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: elastic_net.py

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

__author__ = 'yasaka'

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

elastic_net = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_net.fit(X, y)
print(elastic_net.predict(1.5))

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))




