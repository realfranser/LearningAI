import numpy as np
import math
import matplotlib.pyplot as plt

# Charpter I

## Sigmoid neurons simulating perceptrons, part I
    
inputs = np.array([1,2,3])
weights = np.array([1,1,2])
bias = 1

def perceptron (inputs, weights, bias, threshold):
    value = np.dot(inputs, weights) - bias # Scalar product (inputs X weights)
    return 1 if value > threshold else 0

## Sigmoid neurons simulating perceptrons, part II
def sigmoid (inputs, weights, bias):
    value = -np.dot(inputs, weights)-bias
    return 1/1+math.exp(value)

def first1(c):
    print(perceptron(inputs, weights, bias, 3))
    print(perceptron(inputs*c, weights*c, bias*c, 3))

def first2(c):
    results = []
    perceptron_out = perceptron(inputs, weights, bias, 3)

    for num in range(c, c+1000):
        res = sigmoid(inputs*num, weights*num, bias*num)
        print(res)
        results.append(res)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(results)), results, 'b')
    ax.hlines(y=perceptron_out, xmin=-2, xmax=2, color='r')
    plt.show()
        


def main():
    first1(2)
    first2(2)

if __name__ == '__main__':
    main()
