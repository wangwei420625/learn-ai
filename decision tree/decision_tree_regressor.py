import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
# from tensorflow.contrib.tensor_forest.client import random_forest

N = 100
x = np.random.rand(N) * 6 - 3
x.sort()

y = np.sin(x) + np.random.rand(N) * 0.05
print(y)

x = x.reshape(-1, 1)
print(x)
RandomForestClassifier

dt_reg = DecisionTreeRegressor(criterion='mse', max_depth=9)
dt_reg.fit(x, y)

x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
y_hat = dt_reg.predict(x_test)

plt.plot(x, y, "y*", label="actual")
plt.plot(x_test, y_hat, "b-", linewidth=2, label="predict")
plt.legend(loc="upper left")
plt.grid()
# plt.show()
plt.savefig("./temp_decision_tree_regressor")


# 比较不同深度的决策树
depth = [2, 4, 6, 8, 10]
color = 'rgbmy'
dt_reg = DecisionTreeRegressor()
plt.plot(x, y, "ko", label="actual")
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
for d, c in zip(depth, color):
    dt_reg.set_params(max_depth=d)
    dt_reg.fit(x, y)
    y_hat = dt_reg.predict(x_test)
    plt.plot(x_test, y_hat, '-', color=c, linewidth=2, label="depth=%d" % d)
plt.legend(loc="upper left")
plt.grid(b=True)
# plt.show()
plt.savefig("./temp_compare_decision_tree_depth")

