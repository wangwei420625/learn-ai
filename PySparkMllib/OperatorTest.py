from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf().setMaster("local").setAppName("test")
sc = SparkContext(conf=conf)
rdd1 = sc.parallelize([("zs", 100), ("ls", 50)])
rdd2 = sc.parallelize([("zs", 80), ("zs", 90)])
# data.map(lambda el: el + 1)\
# data.filter(lambda el: el % 2 == 0) \
# data.flatMap(lambda el: [el, el]) \
# data.sample(True, 1.0) \
# data.map(lambda el: [el, el]).groupByKey() \
# data.map(lambda el: [el, el]).reduceByKey(lambd a a, b: a+b) \
# rdd1.union(rdd2) \
# rdd1.join(rdd2) \
# rdd1.mapValues(lambda value: value*2) \
# rdd1.sortBy(lambda kv: kv[1], ascending=False)   \
# rdd1.sortByKey() \
#     .foreach(print)
temp = rdd1.map(lambda kv: kv[1]).reduce(lambda a,b: a+b)
print(temp)



