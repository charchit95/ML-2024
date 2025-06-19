import sys
import numpy as np
import pandas as pd
from activation_functions import *
import csv


def add_specific_n_layers(nn, number, act_fun, num_units):
    """
        Add a specific number of hidden layer to the model
    :param nn: neural network
    :param number: number of hidden layer
    :param act_fun: activation function for hidden layer
    :param num_units: number of units of each hidden layer
    """
    for _ in range(number):
        nn.add_layer(act_fun, num_units)


def der_square_loss(target, prediction):
    """
        Computes the derivative of mean square error
        :param target: target
        :param prediction: prediction of model
        :return errors: error at output layer
    """
    errors = []
    for x, y in zip(target, prediction):
        errors.append(x - y)
    return errors


def mean_square_error(target, prediction):
    """
        Computes the mean square error
        :param target: target
        :param prediction: prediction of model
    """
    loss = 0
    results = []
    n_pattern = np.asarray(prediction).shape[1]
    n_target = np.asarray(prediction).shape[0]
    for k in range(n_pattern):
        if n_target == 1:
            loss += np.square(target[k] - prediction[0][k])
        else:
            for i in range(n_target):
                loss += np.square(target[i][k] - prediction[i][k])
        results.append(loss)
        loss = 0
    return np.mean(results)


def der_square_reg_loss(target, prediction, lambda1, weights):
    """
        Computes the derivative of mean square error with regularization
        :param target: target
        :param prediction: prediction of model
        :param lambda1: lambda value
        :param weights: sum of square of output layer weights
        :return errors: error at output layer with regularization
    """
    loss = []
    for x, y in zip(target, prediction):
        loss.append((x - y) + (lambda1 * weights))
    return loss


def error_reg(target, prediction, lambda1, weights):
    """
        Computes the mean square error with regularization
        :param target: target
        :param prediction: prediction of model
        :param lambda1: lambda value
        :param weights: sum of square of output layer weights
    """
    loss = 0
    results = []
    n_pattern = np.asarray(prediction).shape[1]
    n_target = np.asarray(prediction).shape[0]
    for k in range(n_pattern):
        if n_target == 1:
            loss += np.square(target[k] - prediction[0][k]) + lambda1 * weights
        else:
            for i in range(n_target):
                loss += np.square(target[i][k] - prediction[i][k]) + lambda1 * weights[i]
        results.append(loss)
        loss = 0
    return np.mean(results)


def mean_euclidian_error(target, prediction):
    """
        Computes the mean euclidian error to evaluate the performance of regression task
        :param target: target
        :param prediction: prediction of model
    """
    loss = 0
    results = []
    n_pattern = np.asarray(prediction).shape[1]
    n_target = np.asarray(prediction).shape[0]
    for k in range(n_pattern):
        for i in range(n_target):
            loss += np.square(target[i][k] - prediction[i][k])
        loss = np.sqrt(loss)
        results.append(loss)
        loss = 0
    return np.mean(results)


def confusion_matrix(target, prediction):
    """
        Computes the confusion matrix to calculate the accuracy score
        :param target: target
        :param prediction: prediction of model
        :return true_positive, true_negative, false_positive and false_negative
    """
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    for y, y1 in zip(target, prediction):
        if y == y1:
            if y == 0:
                true_negative += 1
            else:
                true_positive += 1
        else:
            if y == 0:
                false_negative += 1
            else:
                false_positive += 1
    return true_negative, true_positive, false_negative, false_positive


def accuracy(target, prediction):
    """
        Computes the accuracy for classification problem
        :param target: target
        :param prediction: prediction of model
        :return accuracy: percentage of accuracy
        """
    tn, tp, fn, fp = confusion_matrix(target, prediction)
    return ((tp + tn) / (tp + tn + fn + fp)) * 100


def ridge_l2(lambda1, w):
    """
    Computes the derivative of Tikhonov regularization (L2)
    :param lambda1: regularization coefficient
    :param w: the list of each layer's weights
    """
    return 2 * lambda1 * w


def oneHotEncoder(features):
    """
    OneHotEncoder for monks dataset
    :param features: input vector (6 elements)
    :return: encoded input vector (17 elements)
    """
    # num_features is used to memorize how many value a particular feature could take
    num_features = np.array([3, 3, 2, 3, 4, 2])  # attribute value information
    ohe = np.zeros(17)  # prepare an array full of 17 zero where we store the example encoded
    j = 0
    for i in range(len(features)):
        sub = features[i] - 1
        ohe[sub + j] = 1
        j += num_features[i]
    return ohe


def importMonk(dataset):
    """
    Import Monk classification task 's dataset
    :param dataset: dataset 's file
    :return:
    """
    column_name = ['y1', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'id']
    features = pd.read_csv(dataset, sep=" ", names=column_name)
    features.set_index('id', inplace=True)
    target = features['y1'].to_numpy()

    features.drop(columns=['y1'], inplace=True)
    features = features.to_numpy()
    ohe = np.apply_along_axis(oneHotEncoder, 1, features)
    return ohe, target


def importCUP(train_dataset, blind_test_dataset):
    """
    Import Cup regression task 's datasets
    :param train_dataset: dataset 's file of train data
    :param blind_test_dataset: dataset 's file of blind test data
    :return: train set, test set and blind test set
    """
    column_name = ['id', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'x', 'y', 'z']

    features = pd.read_csv(train_dataset, sep=",", names=column_name, skiprows=7)
    features.set_index('id', inplace=True)

    features = features.iloc[np.random.permutation(len(features))]

    features_train = features[0:800].copy()
    features_test = features[800::].copy()

    y_train = features_train[['x', 'y', 'z']].copy()
    y_test = features_test[['x', 'y', 'z']].copy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    features_train.drop(columns=['x'], inplace=True)
    features_train.drop(columns=['y'], inplace=True)
    features_train.drop(columns=['z'], inplace=True)
    x_train = features_train.to_numpy()

    features_test.drop(columns=['x'], inplace=True)
    features_test.drop(columns=['y'], inplace=True)
    features_test.drop(columns=['z'], inplace=True)
    x_test = features_test.to_numpy()

    column_name = ['id', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10']
    dt_test = pd.read_csv(blind_test_dataset, sep=",", names=column_name, skiprows=7)
    dt_test.set_index('id', inplace=True)
    dt_test = dt_test.to_numpy()
    return x_train, y_train, x_test, y_test, dt_test

def importCUP_tot(train_dataset, blind_test_dataset):
    """
    Import Cup regression task 's datasets
    :param train_dataset: dataset 's file of train data
    :param blind_test_dataset: dataset 's file of blind test data
    :return: train set, test set and blind test set
    """
    column_name = ['id', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'x', 'y', 'z']

    features = pd.read_csv(train_dataset, sep=",", names=column_name, skiprows=7)
    features.set_index('id', inplace=True)

    features = features.iloc[np.random.permutation(len(features))]

    y_train = features[['x', 'y', 'z']].copy()
    y_train = y_train.to_numpy()

    features.drop(columns=['x'], inplace=True)
    features.drop(columns=['y'], inplace=True)
    features.drop(columns=['z'], inplace=True)
    x_train = features.to_numpy()

    column_name = ['id', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10']
    dt_test = pd.read_csv(blind_test_dataset, sep=",", names=column_name, skiprows=7)
    dt_test.set_index('id', inplace=True)
    dt_test = dt_test.to_numpy()
    return x_train, y_train, dt_test

def print_results(neural, X_blind_test):
    """
    Predict blind test and write the results into a csv file
    :param neural: final model
    :param X_blind_test: blind test set
    :return:
    """
    # Prediction
    y_pred = neural.predict(np.asarray(X_blind_test))
    with open('results/techies_ML-CUP23-TS.csv', 'w', newline='') as csvfile:
        csvfile.write("# Claudia Muscente, Anson Johnson Madambi \n")
        csvfile.write("# Techies \n")
        csvfile.write('# ML-CUP23 \n')
        csvfile.write('# Submission Date 31/01/2024 \n')

        writer = csv.writer(csvfile)

        for idx, pr in enumerate(np.asarray(y_pred).T):
            csvfile.write(str(idx) + ",")
            writer.writerow(pr)
