import sys
sys.path.append('lib')

import tensorflow as tf
import input_data


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return


if __name__ == '__main__':
    main()
