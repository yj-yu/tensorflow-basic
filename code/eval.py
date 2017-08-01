import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from IPython import embed
from tensorflow import flags

def main(_):
  mnist = input_data.read_data_sets("./data", one_hot=True)

  # defien model input: image and ground-truth label
  model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

  # define parameters for Logistic Regression model
  w = tf.Variable(tf.zeros(shape=[784, 10]))
  b = tf.Variable(tf.zeros(shape=[10]))

  logits = tf.matmul(model_inputs, w) + b
  predictions = tf.nn.softmax(logits)

  # define cross entropy loss term
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=labels,
    logits=predictions)

  dense_predictions = tf.argmax(predictions, axis=1)
  dense_labels = tf.argmax(labels, axis=1)
  equals = tf.cast(tf.equal(dense_predictions, dense_labels), tf.float32)
  acc = tf.reduce_mean(equals)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    final_acc = 0.0
    sess.run(tf.global_variables_initializer())
    for step in range(50):
      images_val, labels_val = mnist.validation.next_batch(100)
      feed = {model_inputs: images_val, labels: labels_val}
      acc = sess.run(acc, feed_dict=feed)
      final_acc += acc
    final_acc /= 50.0
    print ("Full Evaluation Accuracy : {}".format(final_acc))


if __name__ == "__main__":
  tf.app.run()




