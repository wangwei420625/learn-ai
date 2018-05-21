# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn import grid_search
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.mixture import GMM


def expand(a,b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    #生成400个数据
    N = 400
    centers = 4
    #生成了 400个 2维 均方差的随机点， 分布在四个中心点周围
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    #生成 400个 方差不同的数据集，分布在四个中心点周围
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(5, 5, 0.5, 2), random_state=2)
    #构建了一个样本不均衡的数据集
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    print(data3)
    #构建data3对应的标签数组
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    #生成一个kmeans聚类器
    cls = KMeans(n_clusters=4, init='k-means++')
    y_hat = cls.fit_predict(data)

    #分别打印评估分数
    # print(y_hat)
    # print(calinski_harabaz_score(data,y_hat))
    # print(silhouette_score(data,y_hat))
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)


    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)
    y_r_hat = cls.fit_predict(data_r)

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title(u'旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title(u'旋转后KMeans++聚类')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title(u'方差不相等数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title(u'方差不相等KMeans++聚类')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title(u'数量不相等数据')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title(u'数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    # https://github.com/matplotlib/matplotlib/issues/829
    # plt.subplots_adjust(top=0.92)
    plt.show()
    # plt.savefig('cluster_kmeans')


