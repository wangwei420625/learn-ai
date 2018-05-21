from kafka import KafkaProducer

brokers = ['spark01:9092', 'spark02:9092', 'spark03:9092']
producer = KafkaProducer(bootstrap_servers=brokers)
producer.send("car_events", value=b"i love xuruyun")
producer.flush()


