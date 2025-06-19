import numpy as np

from activation_functions import *


class Unit:

    def __init__(self, num, dim_weights, range_weights):
        self.num = num
        range_weights = 0.8
        self.range_weights = np.random.uniform(-range_weights, range_weights,
                                               size=dim_weights)

        self.__weights = np.asarray(self.range_weights)
        self.__net = 0
        self.__value = []
        self.__deltas = []
        self.__deltas_weight = []
        self.__deltas_weight_old = np.zeros((1), dtype=int)  # for momentum
        self.__bias = 0

    def save_value(self, value):
        self.__value = value

    def save_net(self, net):
        self.__net = net

    def save_weight(self, weight):
        self.__weights = np.array(weight)

    def save_deltas(self, deltas):
        self.__deltas = deltas

    def save_deltas_weight(self, deltas_weight):
        self.__deltas_weight = deltas_weight

    def save_deltas_weight_old(self, deltas_weight_old):
        self.__deltas_weight_old = deltas_weight_old

    def save_bias(self, bias):
        self.__bias = bias

    def get_weights(self):
        return self.__weights

    def get_value(self):
        return self.__value

    def get_net(self):
        return self.__net

    def get_bias(self):
        return self.__bias

    def get_deltas(self):
        return self.__deltas

    def get_deltas_weight(self):
        return self.__deltas_weight

    def get_deltas_weight_old(self):
        return self.__deltas_weight_old
