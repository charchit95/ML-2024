import numpy as np
from NN import NN
from utility import *
import random
from activation_functions import *
from tqdm import tqdm

def kfold_indices(X, k):
    """
    Computes indices to divide into k parts the dataset
    :param X: input dataset
    :param k: number of partitions
    :return: list of indices for each partition
    """
    len_X = np.asarray(X).shape[1]
    fold_size = len_X // k

    folds = []
    a = np.arange(len_X)
    b = np.random.permutation(a)

    chunk_size = np.arange(0, len_X, fold_size)
    indices_validation = np.split(b, chunk_size)
    indices_validation = indices_validation[:k + 1]

    for i in range(1, k + 1):
        val_indices = indices_validation[i]
        train_indices = np.setdiff1d(a, indices_validation[i])
        folds.append((train_indices, val_indices))
    return folds


def search_kfold_parameters(X, y, n_epochs, error_rate, k, act_fun, momentum, regularization, regression, *args):
    """
    Apply k-fold cross validation using input parameters, to find possible combination of one parameter
    :param X: Input dataset
    :param y: target dataset
    :param n_epochs: number of epochs
    :param error_rate: error
    :param k: number of partitions
    :param act_fun: activation function
    :param momentum: momentum, true if is activated
    :param regularization: regularization, true if is activated
    :param regression: regression, true if CUP dataset
    :param args: parameters, like eta, alpha, ecc
    :return:
    """
    if regression:
        type_d = 'cup'
    else:
        type_d = 'monk'
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        X = np.asarray(X)
        y = np.asarray(y)

        # Get the fold indices
        fold_indices = kfold_indices(X, k)

        scores = []
        errors = []
        epoch = []
        for train_indices, val_indices in fold_indices:
            if regression:
                X_train, y_train = X.T[train_indices], y.T[train_indices]
                X_val, y_val = X.T[val_indices], y.T[val_indices]
                y_train = y_train.T
                y_val = y_val.T
            else:
                X_train, y_train = X.T[train_indices], y[train_indices]
                X_val, y_val = X.T[val_indices], y[val_indices]

            X_train = X_train.T
            X_val = X_val.T

            # Create a model
            model = NN(X.shape[0], momentum, type_d, regularization)
            add_specific_n_layers(model, prod[0], act_fun, prod[1])

            # Train the model on the training data
            error, e = model.train(n_epochs, error_rate, prod[2], X_train, y_train, prod[3], prod[4])

            # Make predictions on the test data
            y_pred = model.predict(X_val)

            if regression:
                # Calculate the mee score for this fold
                fold_score = mean_euclidian_error(y_val, y_pred)
            else:
                # Calculate the accuracy score for this fold
                fold_score = accuracy(y_val, y_pred)

            # Append the fold score to the list of scores
            scores.append(fold_score)
            errors.append(error[-1])
            epoch.append(e)

        yield prod, errors, epoch, scores


def k_fold_cross_validation(X, y, epochs, error_rate, k, param_model):
    """
    Apply k-fold cross validation
    :param X: input dataset
    :param y: target dataset
    :param epochs: number of epochs
    :param error_rate: error
    :param k: number of partitions
    :param param_model: parameter of the model
    :return: parameter that perform best model and evaluation of the model
    """
    params = {}
    mean_error = 100
    mean_accuracy = 0
    mean_scores = 10000
    keys, values = zip(*param_model.items())
    values = list(values)
    param_search = []
    regularization = values[6]
    momentum = values[4]
    if not regularization:
        values[7] = np.asarray([0])
    if not momentum:
        values[5] = np.asarray([0])
    param_search.append(values[0])
    param_search.append(values[1])
    param_search.append(values[3])
    param_search.append(values[5])
    param_search.append(values[7])

    keys = ('hidden_layers', 'n_units', 'eta', 'alpha', 'lambda')
    stats = list(
        search_kfold_parameters(X, y, epochs, error_rate, k, values[2], values[4], values[6], values[8], *param_search))

    with open("datasets/results_kfold.txt", "w") as my_file:
        for stat in stats:
            my_file.write("mean error: " + str(np.mean(stat[1])) +
                          " mean epochs: " + str(np.mean(stat[2])) +
                          " mean score: " + str(np.mean(stat[3])) +
                          " parameters: " + str(dict(zip(keys, stat[0]))) + "\n")
            if values[8]:  # regression
                if np.mean(stat[3]) <= mean_scores and np.mean(stat[1]) < mean_error:
                    params = dict(zip(keys, stat[0]))
                    mean_scores = np.mean(stat[3])
                    mean_error = np.mean(stat[1])
            else:
                if np.mean(stat[3]) >= mean_accuracy and np.mean(stat[1]) < mean_error:
                    params = dict(zip(keys, stat[0]))
                    mean_accuracy = np.mean(stat[3])
                    mean_error = np.mean(stat[1])
    if not values[8]:
        mean_scores = mean_accuracy
    return params, mean_scores, mean_error


def search_parameters(X_train, y_train, X_test, y_test, n_epochs, error_rate, momentum, regularization, regression,
                      *args):
    """
    Apply grid search to find the best combination of different parameter
    :param X_train: train input dataset
    :param y_train: train target dataset
    :param X_test: test input dataset
    :param y_test: test target dataset
    :param n_epochs: number of epochs
    :param error_rate: error
    :param momentum: momentum, true if is activated
    :param regularization: regularization, true if is activated
    :param regression: regression, true if CUP dataset
    :param args: parameters, like eta, alpha, ecc
    :return: parameters that build the best model
    """
    if regression:
        type_d = 'cup'
    else:
        type_d = 'monk'

    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]

    for prod in tqdm(result):

        neural = NN(np.asarray(X_train).shape[0], momentum, type_d, regularization)
        add_specific_n_layers(neural, prod[0], Tanh, prod[1])
        errors, e = neural.train(n_epochs, error_rate, prod[2], X_train, y_train, prod[3], prod[4])
        y_pred = neural.predict(np.asarray(X_test))
        if regression:
            mee = mean_euclidian_error(y_test, y_pred)
            score = mee
        else:
            acc = accuracy(y_test, y_pred)
            score = acc

        yield prod, score, errors[-1], e


def grid_search(X_train, y_train, X_test, y_test, param_grid, momentum, n_epochs, error_rate, regularization,
                regression):
    """
    Apply grid search to find the best combination of different parameter
    :param X_train: train input dataset
    :param y_train: train target dataset
    :param X_test: test input dataset
    :param y_test: test target dataset
    :param param_grid: parameters of grid search
    :param momentum: momentum, true if is activated
    :param n_epochs: number of epochs
    :param error_rate: error
    :param regularization: regularization, true if is activated
    :param regression: regression, true if CUP dataset
    :return: parameters that build the best model
    """
    params = {}
    error = 100
    acc = 0
    score = 10000
    keys, values = zip(*param_grid.items())
    values = list(values)
    if not regularization:
        values[4] = np.asarray([0])

    if not momentum:
        values[3] = np.asarray([0])

    stats = list(
        search_parameters(X_train, y_train, X_test, y_test, n_epochs, error_rate, momentum, regularization, regression,
                          *values))

    with open("datasets/results_greedy.txt", "w") as my_file:
        for stat in stats:
            my_file.write("accuracy: " + str(stat[1]) +
                          " error: " + str(stat[2]) +
                          " epoch: " + str(stat[3]) +
                          " parameters: " + str(dict(zip(keys, stat[0]))) + "\n")

            if regression:  # regression
                if stat[1] <= score and stat[2] < error:
                    params = dict(zip(keys, stat[0]))
                    score = stat[1]
                    error = stat[2]
            else:
                if stat[1] >= acc and stat[2] < error:
                    params = dict(zip(keys, stat[0]))
                    acc = stat[1]
                    error = stat[2]

    if not regression:
        score = acc
    return params, score, error
