# 共享单车系统，可以在某一点租借，在另一个地方归还，预测不同情况
# 季节、月份、时间、假日、星期、工作日、天气、温度、体感温度、湿度、风速
# 条件下，每一个小时的租用数量

# https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# feature特征、label标签

from pyspark import SparkConf
from pyspark import SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from time import time
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
import pandas as pd
import matplotlib.pyplot as plt
import math


conf = SparkConf().setAppName("DecisionTree").setMaster("local")
sc = SparkContext(conf=conf)


# 提取特征字段
def extract_features(record, featureEnd):
    featureSeason = [convert_float(field) for field in record[2]]
    features = [convert_float(field) for field in record[4: featureEnd - 2]]
    return np.concatenate((featureSeason, features))


def convert_float(x):
    # 很多字段没有数值，是问号，那么给默认值0
    return 0 if x == "?" else float(x)


# 提取label标签字段
def extract_label(record):
    label = record[-1]
    return float(label)


def PrepareData(sc):
    # 1,导入并转换数据
    if sc.master[0:5] == 'local':
        path = "../"
    else:
        path = "hdfs://spark01:9000/"

    print("开始导入数据...")
    rawDataWithHeader = sc.textFile("../data/hour.csv")
    temp = rawDataWithHeader.collect()
    header = temp[0]
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split(","))
    print(" 共计：" + str(lines.count()) + " 项 ")

    # 2,建立训练评估所需数据
    labelpointRDD = lines.map(lambda r:
                              LabeledPoint(
                                  extract_label(r),
                                  extract_features(r, len(r) - 1)
                              ))

    # 3,随机方式将数据分为3个部分并返回
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData:" + str(trainData.count()) +
          " validationData:" + str(validationData.count()) +
          " testData:" + str(testData.count()))

    # 返回数据
    return trainData, validationData, testData


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLables)
    RMSE = metrics.rootMeanSquaredError
    return RMSE


def trainEvaluateModel(trainData, validationData,
                       impurityParam, maxDepthParam, maxBinsParam):
    startTime = time()
    model = DecisionTree.trainRegressor(trainData,
                                        categoricalFeaturesInfo={},
                                        impurity=impurityParam,
                                        maxDepth=maxDepthParam,
                                        maxBins=maxBinsParam)
    RMSE = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数" +
          " impurity=" + str(impurityParam) +
          " maxDepth=" + str(maxDepthParam) +
          " maxBins=" + str(maxBinsParam) + "\n" +
          " ==>所需时间=" + str(duration) +
          " 结果RMSE=" + str(RMSE))
    return RMSE, duration, impurityParam, maxDepthParam, maxBinsParam, model


# 改为RMSE评估参数
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
                      columns=['RMSE', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    # 显示图形
    showchart(df, evalparam, 'RMSE')


# 之前建立的pandas DataFrame可以使用Matplotlib绘图，编写showchart()函数
def showchart(df, evalParam, barData):
    ax = df[barData].plot(kind='bar', title=evalParam,
                          figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalParam, fontsize=12)
    ax.set_ylim([0, 200])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()  # 建立另外一个图形
    ax2.plot(df[['duration']].values, linestyle='-', marker='o', linewidth=2.0, color='r')
    plt.show()


def evalAllParameter(trainData, validationData,
                     impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData,
                                  impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出RMSE最小的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0])
    bestParameter = Smetrics[0]
    # 显示调参后最佳参数
    print("调参后最佳参数：impurity:" + str(bestParameter[2]) +
          ", maxDepth:" + str(bestParameter[3]) +
          ", maxBins:" + str(bestParameter[4]) + "\n" +
          ", 结果AUC = " + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]


def PredictData(sc, model, labelpointRDD):
    # 1, 导入并转换数据
    # 2, 建立预测所需数据LabeledPoint RDD
    # 3, 定义字典
    SeasonDict = {1: '春', 2: '夏', 3: '秋', 4: '冬'}
    HoildayDict = {0: '非假日', 1: '假日'}
    WeekDict = {0: '一', 1: '二', 2: '三', 3: '四', 4: '五', 5: '六', 6: '日'}
    WorkDayDict = {1: '工作日', 0: '非工作日'}
    WeatherDict = {1: '晴', 2: '阴', 3: '小雨', 4: '大雨'}
    # 4, 进行预测并显示结果
    for lp in labelpointRDD.take(20):
        predict = int(model.predict(lp.features))
        label = lp.label
        features = lp.features
        result = "正确" if label == predict else "错误"
        error = math.fabs(label - predict)
        print("==>预测结果：" + str(predict) + " 实际：" + str(label) + " 误差：" + str(error))









