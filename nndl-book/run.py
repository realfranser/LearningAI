import os
import argparse
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('test')
args = parser.parse_args()


def test1():
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def test2():
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

def test3():
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def test4():

    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)


def test5():
    net = network.Network([784, 100, 30, 10])
    net.SGD(training_data, 40, 10, 1.0, test_data=test_data)

globals()[args.test]()


