import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)

X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], marker='o')


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
f2 = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)

from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
f3 =plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)

y_pred = DBSCAN(eps = 0.1).fit_predict(X)
f4 = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

plt.show()
