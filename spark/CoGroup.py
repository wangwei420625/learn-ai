from pyspark import  SparkContext
from pyspark import  SparkConf

conf= SparkConf().setAppName("WordCount").setMaster("local[*]")
context = SparkContext(conf=conf)

context.setLogLevel("INFO")

x=context.parallelize([("a",1),("b",4)])
y=context.parallelize([("a",2)])

[(x,tuple(map(list,y))) for x, y in sorted(list(x.cogroup(y).collect()))]