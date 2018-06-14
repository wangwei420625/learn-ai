import tensorflow as tf


def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_uniform(w_shape), name='weights')
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.maximum(z, 0., name='relu')


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = output.eval(feed_dict={X: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
    print(result)

