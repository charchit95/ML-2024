import numpy as np
import math


class Activation:
    def __init__(self, name: str, func, deriv):
        """
        Utility class used to manage the name, function, and its derivative for all activation functions.
        :param name: Name of the activation function, shown in the title of charts.
        :param func: Pointer to the activation function.
        :param deriv: Pointer to the derivative of the activation function.
        """
        self.name = name
        self.func = func
        self.deriv = deriv


def linear(x):
    return x


def d_linear(x):
    return np.ones(x.shape)


def relu(x):
    r = []
    for i in x:
        r.append(max(0, i))
    return np.asarray(r)


def d_relu(x):
    r = []
    for i in x:
        r.append(0 if i < 0 else 1)
    return np.asarray(r)


from scipy.special import expit

def sigmoid(x):
    #return 1 / (1 + np.exp(-x))
    return expit(x)


def d_sigmoid(x):
    xt = 1 / (1 + np.exp(-x))
    xr = xt * (1 - xt)
    return xr


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return 1 - np.square(f)


# Activation functions
Identity = Activation("identity", linear, d_linear)
ReLu = Activation("ReLu", relu, d_relu)
Sigmoid = Activation("Sigmoid", sigmoid, d_sigmoid)
Tanh = Activation("Tanh", tanh, d_tanh)
