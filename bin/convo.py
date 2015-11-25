import tensorflow as tf
import sys
import pdb

sys.path.append('lib')

import input_data

NUM_CORES = 6
LEARNING_RATE = 0.01
MOMENTUM = 0.01
BATCH_SIZE = 100

config = tf.ConfigProto(
    inter_op_parallelism_threads=NUM_CORES,
    intra_op_parallelism_threads=NUM_CORES
)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

with tf.Session(config=config) as sess:
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = optimizer.minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )