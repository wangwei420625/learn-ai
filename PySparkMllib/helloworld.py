from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel


conf = SparkConf().setAppName("wordcount").setMaster("local")
sc = SparkContext(conf=conf)
sc.setCheckpointDir("./chk")
lines = sc.textFile("./text")
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
pairs.persist(storageLevel=StorageLevel(True, True, False, False, 3))
result = pairs.reduceByKey(lambda a, b: a+b)
result.checkpoint()
sorted_result = result.sortBy(lambda kv: kv[1], False)
# sorted_result = result.map(lambda kv: (kv[1], kv[0])).sortByKey(False) \
#                     .map(lambda kv: (kv[1], kv[0]))
temp = sorted_result.take(3)
# num = result.count()
# temp = result.collect()
print(temp)
# result.saveAsTextFile("./my_result")

