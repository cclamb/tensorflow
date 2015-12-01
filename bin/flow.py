import tensorflow as tf


NUM_CORES = 6

config = tf.ConfigProto(
    inter_op_parallelism_threads=NUM_CORES,
    intra_op_parallelism_threads=NUM_CORES
)


def build_graph():
    lhs = tf.constant(5)
    rhs = tf.constant(6)
    op = tf.mul(lhs, rhs)
    return op


def run(root_operation):
    with tf.Session(config=config) as session:
        result = session.run(root_operation)
    return result


def main():
    result = run(build_graph())
    print "The result is: %d" % result


if __name__ == '__main__':
    main()
