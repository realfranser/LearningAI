import os
import argparse
import mnist_loader
import network
import network2
import time



def test11():
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def test12():
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

def test13():
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def test14():
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)


def test15():
    net = network.Network([784, 100, 30, 10])
    net.SGD(training_data, 40, 10, 1.0, test_data=test_data)

def test16():
    net = network.Network([784, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def test17():
    net = network.Network([784, 10])
    net.SGD(training_data, 40, 10, 1.0, test_data=test_data)

def test21():
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
            monitor_evaluation_accuracy=True)

def test22():
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
            monitor_evaluation_accuracy=True)

def test23():
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 400, 10, 0.5, evaluation_data=test_data,
            monitor_evaluation_accuracy=True, monitor_training_cost=True)
    # training_data[:1000]


if __name__=='__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('test')
    args = parser.parse_args()

    start = time.time()
    globals()[args.test]()
    print("\nNetwork training time(s): {}".format(time.time() - start))


