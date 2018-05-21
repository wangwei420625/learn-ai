from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils


sc = SparkContext("local[2]", "hello_streaming")
ssc = StreamingContext(sc, 10)
# lines = ssc.socketTextStream("spark03", 8888)
topics = ["20180305"]
kafkaParams = {"metadata.broker.list": "spark01:9092,spark02:9092,spark03:9092"}
lines = KafkaUtils.createDirectStream(ssc, topics, kafkaParams)

words = lines.flatMap(lambda line: line[1].split(" "))
pairs = words.map(lambda word: (word, 1))
result = pairs.reduceByKey(lambda a, b: a+b)

result.pprint()
ssc.start()
ssc.awaitTermination()

