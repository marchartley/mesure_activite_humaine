# from sklearn.svm import SVR
# from tensorflow.keras import Sequential
# from keras.layers import Dense,LSTM,Embedding
import json
import pickle
# from sklearn.utils import check_array
import time

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from warnings import filterwarnings

filterwarnings('ignore')


def splitting_data(url, testsize, independent_variables, dependent_variable):
    # data = cleaningdata(url, independent_variables, dependent_variable)
    data_csv = pd.read_csv(url)
    data_csv.dropna(inplace=True)
    data_csv = data_csv[data_csv['walking_speed'] < 2.0]
    data = {
        'X': data_csv[independent_variables],
        'y': data_csv[dependent_variable]
    }
    message = "success"  # data["message"]
    if message == "success":
        X = data["X"]
        y = data["y"]
        X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testsize, random_state=0
        )
        return {
            "message": message,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.values,
            "y_test": y_test.values,
        }
    else:
        return {"message": message}


def NeuralNetwork(url, testsize, independent_variables, dependent_variable, parameters):
    response = splitting_data(url, testsize, independent_variables, dependent_variable)
    message = response["message"]
    if message == "success":
        X_train, X_test, y_train, y_test = (
            response["X_train"],
            response["X_test"],
            response["y_train"],
            response["y_test"],
        )
        sc = StandardScaler()
        sc.fit(np.concatenate([X_train, X_test]))
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        # if parameters is not None:
        #
        #     regressor = MLPRegressor(
        #         activation=parameters["activation_function"],
        #         alpha=parameters["alpha"],
        #         hidden_layer_sizes=(
        #             parameters["hidden_layer_sizes"],
        #             parameters["hidden_nodes"],
        #         ),
        #         max_iter=parameters["max_iterations"],
        #         random_state=0,
        #         solver=parameters["solver"],
        #     )
        #
        #
        # else:
        params = {
            'hidden_layer_sizes': [10, 100, 500], # [10, 40, 100, 200, 500],
            'activation': ['relu', 'tanh'], # ['identity', 'logistic', 'relu', 'tanh'],
            'solver': ['adam'], # ['lbfgs', 'adam', 'sgd'],
            'alpha': [0.001, 0.1], # [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant'], # ['constant', 'invscaling', 'adaptative'],
            'early_stopping': [True], # [True, False]
        }
        regressor = MLPRegressor(
            max_iter=400,
        )

        cv = sklearn.model_selection.GridSearchCV(regressor, params, cv=5, scoring=[
            'r2',
            'neg_mean_squared_error',
            'neg_mean_absolute_error',
            'explained_variance',
            'max_error',
            'neg_median_absolute_error',
        ], refit='neg_mean_squared_error', verbose=3)
        cv.fit(X_train_std, np.ravel(y_train))

        predictions = cv.best_estimator_.predict(X_test_std)
        actual = y_test
        plt.plot(predictions, label="Predictions")
        plt.plot(actual, label="Ground truth")
        return cv


# plt.subplot(1, 2, 1)
# plt.title("Predictions with 'direction'")
# res_with_dir = NeuralNetwork("sequences_resumed.csv", 0.3,
#                              ["period_pir0", "period_pir1", "period_pir2", "period_pir3", "delta_time_0_2",
#                               "delta_time_1_3", "direction"],
#                              ["walking_speed"], None)
# plt.subplot(1, 2, 2)
# plt.title("Prediction without 'direction'")
res_without_dir = NeuralNetwork("sequences_resumed.csv", 0.3,
                                ["period_pir0", "period_pir1", "period_pir2", "period_pir3", "delta_time_0_2",
                                 "delta_time_1_3"],
                                ["walking_speed"], None)

df_res_without_dir = pd.DataFrame(res_without_dir.cv_results_)
# df_res_with_dir = pd.DataFrame(res_with_dir.cv_results_)

df_res_without_dir.to_csv("res_without_dir_speeds_under_2_0.csv")
# df_res_with_dir.to_csv("res_with_dir_speeds_under_2_0.csv")

plt.show()
