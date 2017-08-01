import tensorflow as tf

class DNN(object):
  def create_model(self, model_inputs):
    initializer = tf.zeros
    w1 = tf.Variable(initializer(shape=[784, 128]))
    b1 = tf.Variable(tf.zeros(shape=[128]))

    """
    w2 = tf.Variable(initializer(shape=[100, 100]))
    b2 = tf.Variable(initializer(shape=[100]))

    
    """
    w3 = tf.Variable(initializer(shape=[128, 32]))
    b3 = tf.Variable(tf.zeros(shape=[32]))

    
    w4 = tf.Variable(initializer(shape=[784, 10]))
    b4 = tf.Variable(tf.zeros(shape=[10]))

    h1 = tf.nn.relu(tf.matmul(model_inputs, w1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, w3) + b3)
    """
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
    logits = tf.matmul(h3,  w4) + b4
    """
    logits = tf.matmul(model_inputs, w4) + b4
    predictions = tf.nn.softmax(logits)

    return predictions

