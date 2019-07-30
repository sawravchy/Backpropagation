import numpy as np

w1 = np.array([0.15, 0.2])
w2 = np.array([0.25, 0.3])
w3 = np.array([0.4, 0.45])
w4 = np.array([0.5, 0.55])
x= np.array([0.05,0.10])
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def error(t,o):
  return((t-o)**2).mean()

class Neuron(object):
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)


b1 = 0.35
b2=0.6
lr=0.5
min=0.001
for i in range(200):
    h1 = Neuron(w1, b1)
    h2 = Neuron(w2, b1)
    o1 = Neuron(w3, b2)
    o2 = Neuron(w4, b2)
    y = np.array([h1.feedforward(x), h2.feedforward(x)])
    output = np.array([o1.feedforward(y), o2.feedforward(y)])
    test=np.array([0.90,0.99])
    l = error(test, output)


    # derivative
    d_error_d_w5 = -(test[0] - output[0]) * deriv_sigmoid(output[0]) * y[0]
    d_error_d_w6 = -(test[0] - output[0]) * deriv_sigmoid(output[0]) * y[1]
    d_error_d_w7 = -(test[1] - output[1]) * deriv_sigmoid(output[1]) * y[0]
    d_error_d_w8 = -(test[1] - output[1]) * deriv_sigmoid(output[1]) * y[1]

    d_error_d_w1 = ((-(test[0] - output[0]) * deriv_sigmoid(output[0]) * w3[0]) + (
                -(test[1] - output[1]) * deriv_sigmoid(output[1]) * w4[0])) * (deriv_sigmoid(y[0]) * x[0])
    d_error_d_w2 = ((-(test[0] - output[0]) * deriv_sigmoid(output[0]) * w3[0]) + (
                -(test[1] - output[1]) * deriv_sigmoid(output[1]) * w4[0])) * (deriv_sigmoid(y[0]) * x[1])
    d_error_d_w3 = ((-(test[0] - output[0]) * deriv_sigmoid(output[0]) * w3[1]) + (
                -(test[1] - output[1]) * deriv_sigmoid(output[1]) * w4[1])) * (deriv_sigmoid(y[1]) * x[0])
    d_error_d_w4 = ((-(test[0] - output[0]) * deriv_sigmoid(output[0]) * w3[1]) + (
                -(test[1] - output[1]) * deriv_sigmoid(output[1]) * w4[1])) * (deriv_sigmoid(y[1]) * x[1])

    # update
    w1[0] -= lr * d_error_d_w1
    w1[1] -= lr * d_error_d_w2
    w2[0] -= lr * d_error_d_w3
    w2[1] -= lr * d_error_d_w4
    w3[0] -= lr * d_error_d_w5
    w3[1] -= lr * d_error_d_w6
    w4[0] -= lr * d_error_d_w7
    w4[1] -= lr * d_error_d_w8

    if i % 10 == 0:
        print("\nEpoch:",i)
        print("Error loss: ", l)
        print("Updated weights:")
        print("W1 = ",w1[0])
        print("W2 = ", w1[1])
        print("W3 = ", w2[0])
        print("W4 = ", w2[1])
        print("W5 = ", w3[0])
        print("W6 = ", w3[1])
        print("W7 = ", w4[0])
        print("W8 = ", w4[1])
