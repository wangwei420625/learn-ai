# https://www.kaggle.com/c/stumbleupon/

from pyspark import SparkConf
from pyspark import SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
import pandas as pd
import matplotlib.pyplot as plt


conf = SparkConf().setAppName("DecisionTree").setMaster("local")
sc = SparkContext(conf=conf)

# StumbleUpon（https://www.stumbleupon.com/）是个性化的搜索引擎，会按用户的兴趣和网页
# 评分等记录推荐给你感兴趣的网页，例如新文章、季节菜单、新闻等
# 超过数千万人使用StumbleUpon查找新网页、图片、影片......
# 有些网页内容是暂时性的ephemeral，例如季节菜单、当日股市涨跌新闻等
# 这些文章可能只是在某一段时间会对读者有意义，过了这段时间对读者就没有意义了
# 有些网页内容是长青的evergreen，例如理财观念、育儿知识等，读者会长久对这些文章感兴趣

# 分辨网页是暂时性或者是长青的，对于StumbleUpon推荐网页给用户会有很大帮助
# 例如，读者A买股票，他可能会对当日股市涨跌感兴趣，可是过了一周就对这则新闻没兴趣了
# 如果是理财观念的文章，读者A可能会长久有兴趣，因此公司找来了大数据分析师，负责“网页分类”
# 大数据项目

# 此时机器学习派上用场来了。我们的目标是利用机器学习，通过大量网页数据进行训练来建立一个模型，
# 并使用这个模型去预测网页是属于暂时性的或者长青的内容。这属于二元分类问题

# 1）如何搜集数据？
# StumbleUpon过去以及累积了大量的网页数据
# https://www.kaggle.com/c/stumbleupon/data
# 字段0~2网址、网址ID、样板文字，这些字段与判断网页是暂时性的或长青的关系不大，所以我们会忽略
# 字段3~25是Feature特征字段，数值字段，内容是有关此网页的相关信息，例如网页分类、链接的数目、
# 图像的比例等
# 字段26这是label，具有两个值，1代表长青，0代表暂时性
# 从网上下载数据到项目data目录下

# 2）如何进行数据准备？
# StumbleUpon数据集原始的数据是文本文件，我们必须经过一连串的处理，提取特征字段与标签字段，
# 创建训练所需的数据格式LabeledPoint
# StumbleUpon数据集的原始数据是文本文件，我们必须经过一连串的处理，提取特征字段与标签字段，
# 建立训练所需的数据格式LabeledPoint，并以随机方式按照8：1：1比例把数据分割为3个部分
# trainData、validationData、testData

if sc.master[0:5] == 'local':
    path = "../"
else:
    path = "hdfs://spark01:9000/"

print("开始导入数据...")
print(path + "data/train.tsv")
rawDataWithHeader = sc.textFile("../data/train.tsv")
temp = rawDataWithHeader.collect()
# 查看前两项数据
print(temp[:2])

header = temp[0]
print(header)
# 删除第一行字段名
rawData = rawDataWithHeader.filter(lambda x: x != header)
# 去掉双引号
rData = rawData.map(lambda x: x.replace("\"", ""))
# 获取每一行数据字段
lines = rData.map(lambda x: x.split("\t"))
print(" 共计：" + str(lines.count()) + " 项 ")

# 查看第一项数据
temp = lines.collect()
# 分类特征字段、数值特征字段、label标签字段
print(temp[0][3:])

# 数据集的第3个字段是网页分类，分类特征字段必须转换为数值字段才能够被分类算法使用
# 转换方法是以one hot编码方式进行。如果网页分类有N个分类，就会转换为N个数值字段
# 创建网页分类字典
categoriesMap = lines.map(lambda fields: fields[3])\
    .distinct().zipWithIndex().collectAsMap()
print(len(categoriesMap))
print(type(categoriesMap))
print(categoriesMap)


# 提取特征字段
def extract_features(field, categoriesMap, featureEnd):
    # 提取分类特征字段,field[3]是分类名称字符串
    categoryIdx = categoriesMap[field[3]]
    # 初始化分类特征
    categoryFeatures = np.zeros(len(categoriesMap))
    # 设置类目特征对应位置为1
    categoryFeatures[categoryIdx] = 1
    # 提取数值特征字段
    numericalFeatures = [convert_float(field) for field in field[4: featureEnd]]
    # 返回 分类特征字段 + 数值特征字段
    return np.concatenate((categoryFeatures, numericalFeatures))


def convert_float(x):
    # 很多字段没有数值，是问号，那么给默认值0
    return 0 if x == "?" else float(x)


# 提取label标签字段
def extract_label(field):
    label = field[-1]
    return float(label)


# 建立训练评估所需的数据
# 后续进行decision tree的训练必须提供LabeledPoint格式的数据，所以我们必须先建立LabeledPoint
labelpointRDD = lines.map(lambda r:
                          LabeledPoint(
                              extract_label(r),
                              extract_features(r, categoriesMap, len(r) - 1)
                          ))
# 查看第一项数据
temp = labelpointRDD.collect()
# 分类特征字段、数值特征字段、label标签字段
print(temp[0])
print(temp[1])

# 以随机方式将数据分为3部分并返回
trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
print("将数据分trainData:" + str(trainData.count()) +
          " validationData:" + str(validationData.count()) +
          " testData:" + str(testData.count()))

# 为了加快后续程序的运行效率，暂存在内存中
trainData.persist()
validationData.persist()
testData.persist()


# 3）如何训练模型？
# 我们将执行DecisionTree训练，并且建立模型
model = DecisionTree.trainClassifier(
    trainData, numClasses=2, categoricalFeaturesInfo={},
    impurity="entropy", maxDepth=5, maxBins=5)
# 其中categoricalFeaturesInfo参数是设置分类特征字段的信息，我们采用one hot编码转换了字段
# 所以设置为空的dict{} maxBins,决策树每一个节点最大分支数


# 4）如何使用模型进行预测？
# 建立DecisionTree模型之后，我们可以使用这个模型进行预测test.tsv数据
def PredictData(sc, model, catogoriesMap):
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile("../data/test.tsv")
    temp = rawDataWithHeader.collect()
    header = temp[0]
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print(" 共计：" + str(lines.count()) + " 项 ")

    dataRDD = lines.map(lambda r: (r[0],
                                   extract_features(r, categoriesMap, len(r))))
    DescDict = {
        0: "暂时性网页（ephemeral）",
        1: "长青网页（evergreen）"
    }
    temp = dataRDD.collect()
    for data in temp[:10]:
        predictResult = model.predict(data[1])
        print("网址：" + str(data[0] + "\n" +
              "            ==> 预测:" + str(predictResult) +
                          "说明:" + DescDict[predictResult] + "\n"))


print("============预测数据===========")
PredictData(sc, model, categoriesMap)
# 可以看到网页网址与预测的结果，我们也可以单击选中网址，查看一下预测结果是否合理
# 不过我们不太可能用人来判断决策树模型的准确率，必须有一个科学的方法评估模型的准确率


# 5）如何评估模型的准确率？
# 有了模型之后，必须要有一个标准来评估模型的准确率。在二元分类中我们使用AUC作为评估标准
# MLlib提供了使用BinaryClassificationMetrics计算AUC的方法，计算步骤如下：
# a)建立scoreAndLabels，b)使用它建立BinaryClassificationMetrics，并使用它
score = model.predict(validationData.map(lambda p: p.features))
scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
metrics = BinaryClassificationMetrics(scoreAndLabels)
print("AUC=" + str(metrics.areaUnderROC))


# 因为后续我们评估模型很多，所以编写函数方便我们后续重复使用它来评估模型
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return AUC


# 6）模型的训练参数如何影响准确率？
# 在训练模型时我们会输入不同的参数，其中DecisionTree参数impurity、maxDepth、maxBins的值
# 会影响准确率以及训练所需的时间，我们将以图表显示这些参数值，显示准确率和训练所需的时间
# 编写trainEvaluateModel函数
def trainEvaluateModel(trainData, validationData,
                       impurityParam, maxDepthParam, maxBinsParam):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,
                                         numClasses=2, categoricalFeaturesInfo={},
                                         impurity=impurityParam, maxDepth=maxDepthParam,
                                         maxBins=maxBinsParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数" +
          " impurity=" + str(impurityParam) +
          " maxDepth=" + str(maxDepthParam) +
          " maxBins=" + str(maxBinsParam) + "\n" +
          " ==>所需时间=" + str(duration) +
          " 结果AUC=" + str(AUC))
    return AUC, duration, impurityParam, maxDepthParam, maxBinsParam, model


# 运行trainEvaluateModel
trainEvaluateModel(trainData, validationData, "entropy", 5, 5)

# 评估impurity参数
impurityList = ['gini', 'entropy']
maxDepthList = [10]
maxBinsList = [10]
metrics = [trainEvaluateModel(trainData, validationData,
                              impurity, maxDepth, maxBins)
           for impurity in impurityList
           for maxDepth in maxDepthList
           for maxBins in maxBinsList]
print(metrics)

# 训练评估的结果以图表显示
# 我们将metrics转换为Pandas DataFrame，并用Pandas DataFrame绘图
IndexList = impurityList
df = pd.DataFrame(metrics, index=IndexList,
                  columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
print(df)


# 之前建立的pandas DataFrame可以使用Matplotlib绘图，编写showchart()函数
def showchart(df, evalParam, barData, lineData, yMin, yMax):
    ax = df[barData].plot(kind='bar', title=evalParam,
                          figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalParam, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()  # 建立另外一个图形
    ax2.plot(df[lineData].values, linestyle='-', marker='o', linewidth=2.0, color='r')
    plt.show()


# 直方图代表AUC、折线图代表运行时间，从图中可以看到entropy准确度比gini好一些，但是并没有很大的差别，
# 而且运行时间entropy所需的时间比gini少，所以针对这个数据集，选择entropy参数是不错的选择
showchart(df, 'impurity', 'AUC', 'duration', 0.5, 0.7)


# 之前步骤我们评估'impurity'参数，并且画出参数值对准确率的影响以及训练所需的时间，后面我们还要评估'maxDepth'
# 'maxBins'，所以我们编写evalParameter函数，可以用来评估不同参数
def evalParameter(trainData, validationData, evalparam, impurityList, maxDepthList, maxBinsList):
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData,
                                  impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 设置当前评估的参数
    if evalparam == 'impurity':
        IndexList = impurityList[:]
    elif evalparam == 'maxDepth':
        IndexList = maxDepthList[:]
    elif evalparam == 'maxBins':
        IndexList = maxBinsList[:]
    # 转换为pandas数据框
    df = pd.DataFrame(metrics, index=IndexList,
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    # 显示图形
    showchart(df, 'impurity', 'AUC', 'duration', 0.5, 0.7)


# 使用函数评估maxDepth参数
evalParameter(trainData, validationData, 'maxDepth',
              impurityList=['gini'],
              maxDepthList=[3, 5, 10, 15, 20, 25],
              maxBinsList=[10])
# 从图中可以看到，maxDepth=5，AUC最高，maxDepth越大，所需时间越多，所以5可能是不错的选择

# 使用函数评估maxBins参数
evalParameter(trainData, validationData, 'maxBins',
              impurityList=['gini'],
              maxDepthList=[10],
              maxBinsList=[3, 5, 10, 50, 100, 200])
# 从图中可以看到，maxBins=5，AUC最高，maxBins越大，所需时间越多，所以5可能是不错的选择


# 7）如何找出准确率最高的参数组合？
# DecisionTree参数impurity、maxDepth、maxBins有不同的排列组合，我们将所有参数训练评估找出
# 最好的参数组合
def evalAllParameter(trainData, validationData,
                     impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData,
                                  impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出AUC最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smetrics[0]
    # 显示调参后最佳参数
    print("调参后最佳参数：impurity:" + str(bestParameter[2]) +
          ", maxDepth:" + str(bestParameter[3]) +
          ", maxBins:" + str(bestParameter[4]) + "\n" +
          ", 结果AUC = " + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]


print("所有参数训练评估找出最好的参数组合")
bestModel = evalAllParameter(trainData, validationData,
                             ["gini", "entropy"],
                             [3, 5, 10, 15, 20, 25],
                             [3, 5, 10, 50, 100, 200])

# 8）如何确认是否Overfitting（过拟合）？
# Overfitting是指机器学习所学到的模型过度贴近trainData，从而导致误差变得很大。我们会使用
# 另外一组数据testData再次测试，以避免过拟合的问题。如果训练评估阶段时AUC很高，但是测试阶段
# AUC很低，代表可能有过拟合的问题。如果测试与训练评估阶段的结果中AUC差异不大，就代表没有过拟合
# 首先，在训练评估阶段使用validationData评估模型
# 然后，在测试阶段使用另外一组数据testData测试数据后在测试模型
AUC = evaluateModel(model, testData)
print("AUC=" + str(AUC))



