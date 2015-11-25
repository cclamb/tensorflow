import tensorflow as tf
import sys
import pdb

sys.path.append('lib')

import input_data

NUM_CORES = 6
LEARNING_RATE = 0.01
MOMENTUM = 0.01

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
# optimizer = tf.train.AdagradOptimizer(LEARNING_RATE)
# optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)

config = tf.ConfigProto(
    inter_op_parallelism_threads=NUM_CORES,
    intra_op_parallelism_threads=NUM_CORES
)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder('float', [None, 784])

W = tf.Variable(tf.ones([784, 10]))
b = tf.Variable(tf.ones([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder('float', [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session(config=config) as sess:
    sess.run(init)
    for i in xrange(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print sess.run(
        accuracy,
        feed_dict={x: mnist.test.images, y_: mnist.test.labels}
    )

print 'Finished.'
