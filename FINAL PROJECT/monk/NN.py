import sys

import numpy as np
from Layer import Layer
from activation_functions import *
from utility import *
from warnings import warn


class NN:
    # attributes
    def __init__(self, units, momentum, type_d='monk', regularization=False):
        self.layers = []
        self.momentum = momentum
        self.regularization = regularization
        # create first layer
        layer1 = Layer(units=units, dim_weights_prev=0, type_layer=0)
        self.layers.append(layer1)

        self.output_l = False
        self.type_d = type_d    # indicates the type of dataset: monk or CUP

    def add_layer(self, act_function, units):
        """
        Add new hidden layer to the model
        :param act_function: activation function
        :param units: number of units
        """
        # weights of previous layer
        dim_layer_prev_weight = self.layers[-1].get_dim_units()

        # fan-in for init weights
        range_weights = 1 / np.sqrt(dim_layer_prev_weight)
        #range_weights = 0.2

        layer = Layer(units, dim_layer_prev_weight, act_function, range_weights)
        self.layers.append(layer)

    def output_layer(self, act_function=Sigmoid, units=1):
        """
        Add the output layer to the model
        :param act_function: activation function of output layer
        :param units: number of units (1 for Monk, 3 for CUP)
        """
        if self.type_d == 'cup':
            act_function = Identity
            units = 3
        self.add_layer(act_function, units)

    def forward_pass(self, x):
        """
        Forward pass, for each hidden layer and output layer
        :param x: input
        :return: value of output layer
        """
        out = x
        for layer in self.layers[1::]:
            out = layer.forward_pass(out)
        return out

    def backward(self, y, eta, X, alpha, lambda1=0):
        """
        Backward pass: calculate the pass of backward for the output layer, after
        for the subsequent layers, then adjust all weights of the model
        :param y: target values
        :param eta: parameter for backpropagation
        :param X: input data
        :param alpha: parameter for momentum
        :param lambda1: parameter for regularization
        """
        Do = []
        # calculation of output error, for each unit in the output layer
        # if regularization is selected, the error takes this into account
        for idx, unit in enumerate(self.layers[-1].units):
            if self.type_d == 'cup':
                if self.regularization:
                    error = der_square_reg_loss(y[idx], unit.get_value(), lambda1,
                                                np.sum(unit.get_weights()))
                else:
                    error = der_square_loss(y[idx], unit.get_value())
                Do.append(error * self.layers[-1].act_function.deriv(unit.get_net()))
            else:
                if self.regularization:
                    error = der_square_reg_loss(y, unit.get_value(), lambda1,
                                                np.sum(unit.get_weights()))
                else:
                    error = der_square_loss(y, unit.get_value())
                Do.append(error * self.layers[-1].act_function.deriv(unit.get_net()))

        # Calculation of each delta of the output layer
        delta_w = []
        values = self.layers[-2].layer_values()
        for idx, unit in enumerate(self.layers[-1].units):
            unit.save_deltas(Do[idx])
            for i in range(unit.get_weights().shape[0]):
                delta_w.append(np.dot(unit.get_deltas(), np.asarray(values[i])))
            unit.save_deltas_weight(np.asarray(delta_w))
            delta_w = []

        D = np.asarray(Do)
        # weights per unit, useful for backpropagation fo hidden layers
        weights = []
        for layer in self.layers:
            weights.append(layer.layer_weights_per_unit())

        weights = weights[::-1]

        # values of each unit of each layer, useful for backpropagation step
        values = [X]
        for layer in self.layers[1:]:
            values.append(layer.layer_values())

        # calculate input for hidden layers for backpropagation
        # exclude values of output layer and values of the last hidden layer(used for output layer)
        layers_val_mod = values[:-2]
        layers_v = layers_val_mod[::-1]

        # layers reverse for backpropagation
        layers_mod = self.layers[1:-1]
        layers_m = layers_mod[::-1]

        # backpropagation for hidden layers, starts from the last hidden layer to the first hidden layer
        for idx, layer in enumerate(layers_m):
            D = layer.backward_pass(D, weights[idx], layers_v[idx])

        # adjust each weight of each layer, except the layer of input
        for layer in self.layers[1::]:
            layer.adjust_weights(eta, alpha, np.asarray(X).shape[1], momentum=self.momentum, lambda1=0)

    def train(self, epochs, error_rate, eta, X, y, alpha, lambda1=0):
        """
        Training of the model
        :param epochs: number of epochs
        :param error_rate: error rate for early stopping
        :param eta: value for backpropagation
        :param X: input data
        :param y: target data
        :param alpha: value for momentum
        :param lambda1: value for regularization
        :return: error, num_epochs, mse_list, accuracy_list
        """

        errors = []
        mse_list = []
        accuracy_list = []

        low_error = False
        e = 0

        if self.momentum:
            if alpha is None:
                warn("Unspecified alpha value")
                alpha = 0  # momentum not active

        # Add output layer if not already added
        if not self.output_l:
            self.output_layer()
            self.output_l = True

        while e < epochs and not low_error:
            self.forward_pass(X)

            # Gather output prediction
            prediction = [unit.get_value() for unit in self.layers[-1].units]

            # Compute MSE
            error = mean_square_error(np.asarray(y), prediction)
            mse_list.append(error)

            # Compute Accuracy
            y_pred = self.predict(np.asarray(X))
            acc = accuracy(y, y_pred)
            accuracy_list.append(acc)

            # Early stopping
            if e > 0 and np.abs(errors[-1] - error) < error_rate:
                low_error = True
            else:
                self.backward(y, eta, X, alpha)

            e += 1
            errors.append(error)

        return errors, e, mse_list, accuracy_list



    def predict(self, X_test):
        """
        Prediction of values from test set, using the model just created.
        If the prediction is on Monk dataset, each value is mapped into 0 or 1
        :param X_test: input of test set
        :return: prediction of the model
        """
        pred = []
        prediction = 0
        pred = self.forward_pass(X_test)
        if self.type_d == 'monk':
            prediction = list(map(lambda x: 1 if x > 0.5 else 0, np.asarray(pred).T))
        else:
            prediction = pred
        return prediction

    def sum_weights(self):
        """
        Sum of weights of the output layer, useful for regularization
        :return:
        """
        total_weights = 0
        layer_out = self.layers[-1]
        for unit in layer_out.get_units():
            for weight in unit.get_weights():
                total_weights += (weight ** 2)
        return total_weights

    def print_NN_1(self):
        for layer in self.layers:
            print(layer)
            print("number of units:", layer.dim_units)
            for unit in layer.units:
                print("unit", unit.num)
                print("weights ", unit.get_weights())
                print("bias", unit.get_bias())

    def print_NN_2(self):
        for layer in self.layers:
            print(layer)
            print("number of units:", layer.dim_units)
            for unit in layer.units:
                print("unit", unit.num)
                print("weights ", unit.get_weights())
                print("bias", unit.get_bias())
                print("net", unit.get_net())
                print("value ", unit.get_value())
                print("delta", unit.get_deltas())
                print("delta weights", unit.get_deltas_weight())

    def print_NN_state(self, e, error):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}")
            if layer.dim_units > 0:
                for idxu, unit in enumerate(layer.units):
                    print(f"\tUnit {idxu}, in  Layer {idx}")
                    print(f"\t\tweights: {np.asarray(unit.get_weights())}")
                    print(f"\t\tdeltas: {np.asarray(unit.get_deltas())}")
                    print(f"\t\tdeltaw: {np.asarray(unit.get_deltas_weight())}")
                    print(f"\t\tnet: {np.asarray(unit.get_net())}")
                    print(f"\t\tval: {np.asarray(unit.get_value())}")
                    print(f"\t\tbias: {unit.get_bias()}")
            else:
                print(f"\tInput layer {idx}")
