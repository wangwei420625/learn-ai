from pyspark import SparkContext
from pyspark.streaming import StreamingContext


sc = SparkContext("local[2]", "transform")

black_list = [("zs", 100), ("ls", 50)]
black_list_rdd = sc.parallelize(black_list)

ssc = StreamingContext(sc, 20)
data = ssc.socketTextStream("spark03", 8888)
# name log
result = data.map(lambda line: tuple(line.split(" "))) \
    .transform(lambda rdd: rdd.leftOuterJoin(black_list_rdd)) \
    .filter(lambda x: x[1][1] is None) \
    .map(lambda x: "%s %s" % (x[0], x[1][0]))

result.pprint()

ssc.start()
ssc.awaitTermination()
