# spark机器学习工作流程ML Pipeline的原理就是将机器学习的每一个阶段
# 例如数据处理、进行训练与测试、建立Pipeline流程形成机器学习工作流程

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
import pyspark.sql.types
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import RandomForestClassifier


conf = SparkConf().setAppName("DecisionTree").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# 一，建立pipeline
# 建立机器学习流程pipeline包含4个阶段stages，前3个阶段是数据处理
# 第4个阶段是DesionTreeClassifier机器学习分类算法

# StringIndexer：将文字的分类特征字段转换为数字
# OneHotEncoder：将一个数字的分类特征字段转为多个字段
# VectorAssembler：将所有的特征字段整合成一个Vector字段
# DesionTreeClassifier：进行训练并且产生模型

# 二，训练
# 训练数据DataFrame使用pipeline.fit()进行训练
# 系统会按照顺序执行每一个阶段，最后产生pipelineModel模型，pipelineModel和pipeline类似
# 只是多了训练后建立的模型Model

# 三，预测
# 新数据DataFrame使用pipelineModel.transform()进行预测
# 系统会按照顺序执行每一个阶段，并使用Decision Tree Classifier Model进行预测，预测完成后，
# 会产生预测结果DataFrame

# DataFrame：Spark ML机器学习API处理的数据格式是DataFrame，我们必须使用数据框存储训练数据，
# 最后预测结果也是数据框，我们可以使用sqlContext读取文本文件创建DataFrame，或将RDD转为DataFrame
# 也可以使用Spark SQL来操作，数据框可以存储不同的数据类型，文字，特征字段所创建的向量Vectors，
# label标签字段

# Transformer：是一个模型，可以使用transform方法将一个DataFrame转换为另一个DataFrame

# Estimator：是一个算法，可以使用fit方法传入DataFrame，产生一个Transformer

# Pipeline：可以串联多个Transformers与Estimators建立ML机器学习的workflow工作流程

# Parameter：Transformers与Estimators都可以共享相同的Parameter API

# 1,数据准备
if sc.master[0:5] == 'local':
    path = "../"
else:
    path = "hdfs://spark01:9000/"

row_df = sqlContext.read.format('csv').option("header", "true")\
    .option("delimiter", "\t").load(path+"data/train.tsv")
print(row_df.count())
row_df.printSchema()
# 查看前10项数据
row_df.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news',
              'label').show(10)


# 我们编写UDF将"?"转换为0
# 编写用户自定义函数
def replace_question(x):
    return "0" if x == "?" else x


# 使用udf方法将replace_question转换为DataFrames UDF用户自定义函数，相当于注册
replace_question = udf(replace_question)

# 下面把row_df数据框第4个字段至最后一个字段转换为double，其中最右是label，其他是features
df = row_df.select(
    ['url', 'alchemy_category'] +
    # col()按照列名取字段数据，alias把别名设置为原来的字段名
    [replace_question(col(column)).cast("double").alias(column)
     for column in row_df.columns[4:]]
)
df.printSchema()
df.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news',
              'label').show(10)

# 将原始数据分成7：3比例的训练数据和测试数据，并且cache()暂存在内存中，以加快后续程序速度
train_df, test_df = df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache()

# 2, StringIndexer：将文字的分类特征字段转换为数字
# StringIndexer的功能类似于categoriesMap，是网页分类特征字段的字典
# StringIndexer是一个Estimator，所以使用上必须分为两个步骤，使用fit产生Transformer
# 然后使用transform方法将一个DataFrame变为另一个DataFrame
categoryIndexer = StringIndexer(
    inputCol='alchemy_category',
    outputCol='alchemy_category_Index'
)
categoryTransformer = categoryIndexer.fit(df)
for i in range(0, len(categoryTransformer.labels)):
    print(str(i) + ':' + categoryTransformer.labels[i])

df1 = categoryTransformer.transform(train_df)
# 查看全部字段名，可以发现新增了'alchemy_category_Index'字段
print(df1.columns)
# 比较转换前后的差异
df1.select('alchemy_category', 'alchemy_category_Index').show(10)

# 3, OneHotEncoder可以将一个数值的分类特征字段转换为多个字段的Vector
encoder = OneHotEncoder(dropLast=False,
                        inputCol='alchemy_category_Index',
                        outputCol='alchemy_category_IndexVec')
# OneHotEncoder使用transform转换后结果是df2
df2 = encoder.transform(df1)
print(df2.columns)

# 4, VectorAssembler可以将多个特征字段整合成一个特征的Vector
# 字段名先加上
assemblerInputs = ['alchemy_category_IndexVec'] + row_df.columns[4:-1]
print(assemblerInputs)
assembler = VectorAssembler(
    inputCols=assemblerInputs,
    outputCol='features'
)
df3 = assembler.transform(df2)
print(df3.columns)

# 查看特征字段
df3.select('features').show(5)
# 可以看到本质上存储的是SparseVector类型
print(df3.select('features').take(1))

# 5, 使用DecionTreeClassifier二元分类
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",
                            impurity='gini', maxDepth=10, maxBins=14)
dt_model = dt.fit(df3)
print(dt_model)
dt4 = dt_model.transform(df3)

# 6, 建立pipeline
pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, dt])
print(pipeline.getStages())

# 7, 使用pipeline进行数据处理与训练
# 因为训练数据执行pipeline的所有阶段，所以会花时间比较长，最后产生的结果是pipelineModel
pipelineModel = pipeline.fit(train_df)
print(pipelineModel.stages[3])

# 我们还可以进一步使用toDebugString查看决策树模型的规则
print(pipelineModel.stages[3].toDebugString)

# 8, 使用pipelineModel进行预测
predicted = pipelineModel.transform(test_df)
# 查看预测后的Schema，发现新增了3个字段
print(predicted.columns)
predicted.select('url', 'features', 'rawprediction', 'probability', 'label', 'prediction').show(10)

# 9, 模型评估准确率
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',
                                          labelCol='label',
                                          metricName='areaUnderROC')
predictions = pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

# 10, 使用TrainValidation进行训练验证找出最佳模型
# MLlib中必须自行编写evalAllParameter函数才能进行训练评估，以便找出最佳参数模型
# 然而，ML中我们可以使用TrainValidation模型进行训练验证找出最佳模型
paramGrid = ParamGridBuilder().addGrid(dt.impurity, ['gini', 'entropy'])\
    .addGrid(dt.maxDepth, [5, 10, 15]).addGrid(dt.maxBins, [10, 15, 20])\
    .build()
tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator,
                           estimatorParamMaps=paramGrid, trainRatio=0.8)
# 建立的pipeline阶段与之前大致相同，只有最后一个阶段tvs是不同的
tvs_pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, tvs])
# 训练和验证
tvs_pipelineModel = tvs_pipeline.fit(train_df)

bestModel = tvs_pipelineModel.stages[3].bestModel
print(bestModel)
print(bestModel.toDebugString[:500])

predictions = tvs_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

# 11, 我们还可以更进一步用crossValidation交叉验证找出最佳模型
# K折交叉验证可以得到可靠稳定的模型，减少过度学习或学习不足的情况，一般常用10折，K越大，效果越好，只是所需时间
# 越长，为了避免等待时间太久，我们只用k=3来说明和示范
cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
cv_pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, cv])
cv_pipelineModel = cv_pipeline.fit(train_df)
bestModel = cv_pipelineModel.stages[3].bestModel
print(bestModel)

predictions = cv_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

# 12, 使用随机森林分类器
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
rf_pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, rf])

rf_pipelineModel = rf_pipeline.fit(train_df)
rf_predicted = rf_pipelineModel.transform(test_df)
evaluator.evaluate(rf_predicted)

paramGrid = ParamGridBuilder().addGrid(rf.impurity, ['gini', 'entropy'])\
    .addGrid(rf.maxDepth, [5, 10, 15]).addGrid(rf.maxBins, [10, 15, 20])\
    .addGrid(rf.numTrees, [10, 20, 30])\
    .build()
rf_tvs = TrainValidationSplit(estimator=rf, evaluator=evaluator,
                              estimatorParamMaps=paramGrid, trainRatio=0.8)
rf_tvs_pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, rf_tvs])
# 训练和验证
rf_tvs_pipelineModel = rf_tvs_pipeline.fit(train_df)

predictions = rf_tvs_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

rf_cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
rf_cv_pipeline = Pipeline(stages=[categoryIndexer, encoder, assembler, rf_cv])
rf_cv_pipelineModel = rf_cv_pipeline.fit(train_df)

predictions = rf_cv_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

DescDict = {
        0: "暂时性网页（ephemeral）",
        1: "长青网页（evergreen）"
    }
for data in predictions.select('url', 'prediction').take(5):
    print("网址：" + str(data[0] + "\n" +
                      "            ==> 预测:" + str(data[1]) +
                      "说明:" + DescDict[data[1]] + "\n"))
