import numpy as np

from NN import NN
from activation_functions import *
from utility import *
import matplotlib.pyplot as plt
from validation import *

if __name__ == '__main__':

    # Create directory for plots
    # dir_name = "../plots/"

    # Training and test set - monk3 -
    ds_monk3 = "monks-3.train"

    ts_monk3 = "monks-3.test"

    # Read monk3 {train - test}
    dataset, y_train = importMonk('datasets/' + ds_monk3)
    dt_test, y_test = importMonk('datasets/' + ts_monk3)

    X_train = []
    X_test = []

    for i in range(dataset.shape[1]):
        X_train.append(dataset[:, i])

    for i in range(dt_test.shape[1]):
        X_test.append(dt_test[:, i])

    # Model parameters
    input_unit = 17
    num_hidden_layer = 1
    num_units_hidden = 4
    momentum = True
    regularization = True
    act_function = ReLu
    num_epochs = 8000
    error_rate = 0.000000001
    eta = 0.75
    #eta = 0.95
    alpha = 0.75
    #alpha = 0.65
    lambda1 = 0.005

    print("Monk 3")
    for i in range(10):

        # Model
        neural = NN(input_unit, momentum, 'monk', regularization)
        add_specific_n_layers(neural, num_hidden_layer, act_function, num_units_hidden)

        errors, e, mse_list, acc_list = neural.train(num_epochs, error_rate, eta, X_train, y_train, alpha)

        # Prediction
        y_pred = neural.predict(np.asarray(X_test))
        acc = accuracy(y_test, y_pred)

        # plot of errors
        # list2 = np.arange(e)
        # plt.title("Errors - Iteration: " + str(i) + "Monk3")
        # plt.plot(list2, errors, label="Train")
        # plt.legend()
        # plt.show()

        # print("\nPrediction:", y_pred)
        # print(y_test)

        print("Accuracy: ", acc, "%")


        # Plot MSE over epochs
        plt.figure()
        plt.title(f"MSE - Monk3")
        plt.plot(np.arange(e), mse_list, color='orange', label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.xlim(left=0)
        plt.legend()
        plt.show()

        # Plot Accuracy over epochs
        plt.figure()
        plt.title(f"Accuracy - Monk3")
        plt.plot(np.arange(e), acc_list, color='green', label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.xlim(left=0)
        plt.legend()
        plt.show()

        for idxl, layer in enumerate(neural.layers):
            print(f"Layer {idxl}")
            for idxu, unit in enumerate(layer.units):
                print(f"\tUnit {idxu} range_weigth: {unit.range_weights}")

    # param_model = {
    #     'hidden_layers': [1],
    #     'n_units': [4],
    #     'act_fun': ReLu,
    #     'eta': [0.75],
    #     'momentum': True,
    #     'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0, 99],
    #     'regularization': True,
    #     'lambda': [0],
    #     'regression': False
    # }

    # par, acc, err = k_fold_cross_validation(X_train, y_train, num_epochs, error_rate, 5, param_model)
    # print(par, acc, err)

    # # Grid Search
    # param_grid = {
    #     'hidden_layers': np.arange(1, 2),
    #     'n_units': np.arange(2, 5),
    #     'eta': np.arange(0.5, 1, 0.05),
    #     'alpha': np.arange(0.5, 1, 0.05),
    #     'lambda': np.arange(0.001, 0.009, 0.001)
    # }

    # params, acc, err = grid_search(X_train, y_train, X_test, y_test, param_grid, momentum, num_epochs, error_rate,
    #                                regularization, False)
    # print("accura: " + str(acc) + ",error: " + str(err) + ",params: " + str(params))

