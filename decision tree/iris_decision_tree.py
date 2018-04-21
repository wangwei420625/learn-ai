import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
# 读取数据集
data = pd.DataFrame(iris.data)

data.columns = iris.feature_names
data['Species'] = load_iris().target
print(data)
print(data.shape)
x = data.iloc[:, :4]  # 花萼长度和宽度
# x = data.iloc[:, :4]  # 花萼长度和宽度
y = data.iloc[:, -1]
# print(type(x),1)
# y = pd.Categorical(data[4]).codes
# print(x)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=6, criterion='entropy')
tree_clf.fit(x_train, y_train)
y_test_hat = tree_clf.predict(x_test)
print("acc score:", accuracy_score(y_test, y_test_hat))


# export_graphviz(
#     tree_clf,
#     out_file="./iris_tree.dot",
#     feature_names=iris.feature_names[:2],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )

# ./dot -Tpng ~/PycharmProjects/mlstudy/bjsxt/iris_tree.dot -o ~/PycharmProjects/mlstudy/bjsxt/iris_tree.png

# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))
RandomForestClassifier
depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    # 生成一个长度为验证集数量的数组，每一个元素是yhat和y是否相等的结果，
    print(list(result))
    if d == 1:
        print(result)
    err = 1 - np.mean(result)
    print(100 * err)
    err_list.append(err)
    print(d, ' 错误率：%.2f%%' % (100 * err))
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度和过拟合', fontsize=18)
plt.grid(True)
plt.show()


# tree_reg = DecisionTreeRegressor(max_depth=2)
# tree_reg.fit(X, y)

