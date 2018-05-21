from pyspark import  SparkContext
from pyspark import  SparkConf

conf= SparkConf().setAppName("WordCount").setMaster("local[*]")
context = SparkContext(conf=conf)

context.setLogLevel("INFO")

lines = context.textFile("./hello_spark.py")

words = lines.flatMap(lambda line: line.split())
pairs = words.map(lambda word: (word, 1))
result = pairs.reduceByKey(lambda a, b: a + b)

result.foreach(print)


