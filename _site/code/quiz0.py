import tensorflow as tf
import numpy as np
a = tf.placeholder(shape=[3,4], dtype=tf.float32)
b = tf.placeholder(shape=[4,6], dtype=tf.float32)

c = tf.matmul(a,b)

sess = tf.Session()

feed = {a:np.random.randn(3,4), b:np.random.randn(4,6)}
result = sess.run(c, feed_dict=feed)

print (result)
