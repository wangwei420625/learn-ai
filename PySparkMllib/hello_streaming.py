from pyspark import SparkContext
from pyspark.streaming import StreamingContext


def update_func(new_values, previous_count):
    if previous_count is None:
        previous_count = 0
    return sum(new_values, previous_count)


sc = SparkContext("local[2]", "hello_streaming")
ssc = StreamingContext(sc, 10)
ssc.checkpoint("./chk")
lines = ssc.socketTextStream("spark03", 8888)

words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
# result = pairs.reduceByKey(lambda a, b: a+b)
result = pairs.updateStateByKey(update_func)

result.pprint()
ssc.start()
ssc.awaitTermination()

