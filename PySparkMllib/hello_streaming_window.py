from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "window")
ssc = StreamingContext(sc, 5)
ssc.checkpoint("hdfs://spark01:9000/my_checkpoint")

lines = ssc.socketTextStream("spark03", 8888)
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
# windowDuration是每次计算的时候算多少数据量
# slideDuration是隔多长时间计算一次
result = pairs.reduceByKeyAndWindow(lambda a, b: a + b, lambda a, b: a - b
                                    , windowDuration=20, slideDuration=10)

result.pprint()

ssc.start()
ssc.awaitTermination()
