import numpy as np
from activation_functions import *
from Unit import Unit
from utility import *


class Layer:

    def __init__(self, units, dim_weights_prev, act_function=None, range_weights=None, type_layer=1):
        if range_weights is None:
            range_weights = 0.1
        if act_function is None:
            act_function = Identity
        self.act_function = act_function
        self.dim_units = units  # number of units per layer
        self.units = []

        self.range_weights = range_weights
        # add weights vector for each unit only for hidden and output units
        if type_layer != 0:
            for u in range(self.dim_units):
                new_unit = Unit(u, dim_weights_prev, self.range_weights)
                self.units.append(new_unit)

    def forward_pass(self, x):
        """
        Computes the forward pass for each layer
        :param x: input of previous layer
        :return: input for the next layer
        """
        return_list = []
        for unit in self.units:
            net = np.dot(unit.get_weights(), x) + unit.get_bias()
            unit.save_net(net)
            unit.save_value(self.act_function.func(net))
            return_list.append(unit.get_value())
        return return_list

    def backward_pass(self, delta_prev, weights_layer_next, values):
        """
        Computes the backward pass for each layer,
        1. calculate the error of the current layer
        2. calculate the derivative of the activation function
        3. calculate the new delta per each unit of the layer
        :param delta_prev: delta of the next layer
        :param weights_layer_next: weights of the next layer
        :param values: values of previous layer
        :return: deltas for the next iteration of backward
        """
        return_list = []
        for idx, unit in enumerate(self.units):
            errore = np.dot(np.asarray(weights_layer_next).T[:, idx], delta_prev)
            der = self.act_function.deriv(unit.get_net())

            delta_new = errore * der
            unit.save_deltas(delta_new)

            delta_w = []
            for i in range(unit.get_weights().shape[0]):
                delta_w.append(np.dot(unit.get_deltas(), np.asarray(values[i])))
            unit.save_deltas_weight(np.asarray(delta_w))

            return_list.append(delta_new)
        return return_list

    def adjust_weights(self, eta, alpha, l, momentum: False, lambda1=0):
        """
        Adjust weights for each layer, calculate of deltas for each weight and adjust weights and bias
        If momentum is selected, the delta weight depends on the previous delta
        :param eta: parameter for calculate the new delta weight
        :param alpha: parameter for momentum
        :param l: size of dataset
        :param momentum: indicates whether momentum is selected
        :param lambda1: parameter for regularization
        """
        if momentum:
            for unit in self.units:
                u_delta = np.mean(unit.get_deltas())

                # delta w_new
                unit.save_deltas_weight(np.float64(np.asarray((eta / l * unit.get_deltas_weight()) +
                                                              (alpha * unit.get_deltas_weight_old()))))
                # delta w_old for next step
                unit.save_deltas_weight_old(unit.get_deltas_weight())
                unit.save_weight(unit.get_weights() + unit.get_deltas_weight() - ridge_l2(lambda1, unit.get_weights()))
                unit.save_bias(unit.get_bias() + (eta / l * u_delta))
        else:
            for unit in self.units:
                u_delta = np.mean(unit.get_deltas())
                # delta w_new
                unit.save_deltas_weight(eta / l * unit.get_deltas_weight())
                unit.save_weight(unit.get_weights() + unit.get_deltas_weight() - ridge_l2(lambda1, unit.get_weights()))
                unit.save_bias(unit.get_bias() + (eta / l * u_delta))

    def layer_weights_per_unit(self):
        """
        :return: an array of array, each array contain the weights connect to a previous unit
        previous unit: a unit of the previous layer
        """
        weights_per_unit = []
        for unit in self.units:
            weights_per_unit.append(unit.get_weights())
        w = list((zip(*weights_per_unit)))
        return w

    def layer_values(self):
        """
            :return: values of a generic hidden layer
        """
        values_per_layer = []
        for unit in self.units:
            values_per_layer.append(unit.get_value())
        return values_per_layer

    def get_units(self):
        return self.units

    def get_dim_units(self):
        return self.dim_units
