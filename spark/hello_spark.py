from pyspark import  SparkContext
from pyspark import  SparkConf


conf = SparkConf().setAppName("test").setMaster("local[1]")
sc = SparkContext(conf=conf)


data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)
print(distData.collect())
