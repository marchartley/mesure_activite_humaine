# from main_Preparingdata import cleaningdata
import math
import random

from sklearn.svm import SVR
# from tensorflow.keras import Sequential
# from keras.layers import Dense,LSTM,Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
)

# from sklearn.utils import check_array
from sklearn import preprocessing
import time
import pickle
import numpy as np

import datetime
import json
import numpy as np
import requests
import urllib.request
import jsonpickle
import pickle
from sklearn import preprocessing
# from main_Preparingdata import cleaningdata
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.metrics import r2_score
import itertools as IT
import pathlib
import types
from matplotlib import pyplot as plt


def abline(slope, intercept=0):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.concatenate([np.array(axes.get_xlim()), np.array(axes.get_ylim())])
    x_vals.sort()
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def balance_dataset(data: pd.DataFrame):
    ws = data
    nb_classes = 10
    mini, maxi = ws['walking_speed'].min(), ws['walking_speed'].max()
    class_width = (maxi - mini) / nb_classes
    classes = []
    for i in range(nb_classes):
        classes.append(ws[np.logical_and(mini + i * class_width <= ws['walking_speed'],
                                         ws['walking_speed'] < mini + (i + 1) * class_width)])
    biggest_class_size = max([len(c) for c in classes])
    median_size = int(np.median([len(c) for c in classes]))

    for i in range(nb_classes):
        # if len(classes[i]) < median_size:
        #     classes[i] = pd.concat([classes[i]] * math.ceil(median_size / len(classes[i])))
        # classes[i] = classes[i].sample(n = median_size)

        classes[i] = pd.concat([classes[i]] * math.ceil(biggest_class_size / len(classes[i])))

    print([len(c) for c in classes])
    # print(classes[-1])
    return pd.concat(classes)


def cleaningdata(url, independent_variables, dependent_variable, data_multiplier=1):
    df = pd.read_csv(url).dropna()
    df = df[np.logical_and(df['walking_speed'] < 3, df['direction'] != 0)]
    # df = balance_dataset(df)
    data_multiplier = int(data_multiplier)
    return {
        'message': 'success',
        'X': pd.concat([df[independent_variables]] * data_multiplier, ignore_index=True).applymap(
            lambda e: e * (1 + random.random() * 0.5 - 0.25)),
        'y': pd.concat([(df[dependent_variable] if dependent_variable is not None else [])] * data_multiplier,
                       ignore_index=True)
    }


def splitting_data(url, testsize, independent_variables, dependent_variable, data_multiplier=1):
    data = cleaningdata(url, independent_variables, dependent_variable, data_multiplier)
    message = data["message"]
    if message == "success":
        X = data["X"]
        y = data["y"]
        X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testsize
        )
        return {
            "message": message,
            "X_train": X_train,
            "X_test": X_test,
            "X": X,
            "y_train": y_train,
            "y_test": y_test,
            "y": y
        }
    else:
        return {"message": message}


def NeuralNetwork(url, testsize, independent_variables, dependent_variable, parameters, data_multiplier=1):
    response = splitting_data(url, testsize, independent_variables, dependent_variable,
                              data_multiplier=1)  # data_multiplier)
    message = response["message"]
    if message == "success":
        X_train, X_test, X, y_train, y_test, y = (
            response["X_train"],
            response["X_test"],
            response["X"],
            response["y_train"],
            response["y_test"],
            response["y"]
        )
        sc = StandardScaler()
        sc.fit(X)

        X_train = pd.concat([pd.DataFrame(X_train)] * data_multiplier)
        y_train = pd.concat([pd.DataFrame(y_train)] * data_multiplier)

        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        if parameters != None:

            regressor = MLPRegressor(
                activation=parameters["activation_function"],
                alpha=parameters["alpha"],
                hidden_layer_sizes=(
                    parameters["hidden_layer_sizes"],
                    parameters["hidden_nodes"],
                ),
                max_iter=parameters["max_iterations"],
                random_state=0,
                solver=parameters["solver"],
            )


        else:
            regressor = MLPRegressor(
                max_iter=200,
                hidden_layer_sizes=40,
                activation="relu",
                random_state=50,
                alpha=0.1,
                solver='adam',
                learning_rate='adaptive',
                shuffle=True

            )

        time1 = time.time()
        regressor.fit(X_train_std, np.ravel(y_train))
        time2 = time.time()
        print("duration of fitting the model:", (time2 - time1) / 60)
        y_predict = abs(regressor.predict(X_test_std))

        y_actual = abs(y_test)
        r_squared_score = r2_score(np.ravel(y_actual), np.ravel(y_predict))
        mse = mean_squared_error(np.ravel(y_actual), np.ravel(y_predict))
        rmse = mean_squared_error(
            np.ravel(y_actual), np.ravel(y_predict), squared=False
        )
        mae = mean_absolute_error(np.ravel(y_actual), np.ravel(y_predict))
        variance = explained_variance_score(np.ravel(y_actual), np.ravel(y_predict))
        maximum_error = max_error(np.ravel(y_actual), np.ravel(y_predict))
        mediane = median_absolute_error(np.ravel(y_actual), np.ravel(y_predict))
        msle = mean_squared_log_error(np.ravel(y_actual), np.ravel(y_predict))
        try:
            mpd = mean_poisson_deviance(np.ravel(y_actual), np.ravel(y_predict))
            mgd = mean_gamma_deviance(np.ravel(y_actual), np.ravel(y_predict))
        # mtd = mean_tweedie_deviance(np.ravel(y_actual), np.ravel(y_predict))
        except Exception as e:
            mpd = None
            mgd = None
        # mtd = None
        tuple_objects = (regressor, X_train, y_train, rmse)
        print("variance", variance)
        print("mediane", mediane)
        print("mse", mse)
        print("rmse", rmse)
        print("max error", maximum_error)

        pickle.dump(tuple_objects, open("NeuralNetworkModel", "wb"))
        resultBytes = pickle.dumps(tuple_objects)

        # np.reshape(y_predict,(20,1))

        plt.plot(y_actual.reset_index(drop=True))

        Y_ac = (y_actual)
        Y_pr = (y_predict)

        plt.plot(y_predict)
        plt.show()
        return resultBytes, Y_ac, Y_pr
    # else:
    #     return message


# le = preprocessing.LabelEncoder()


# def myconverter(o):
#     if isinstance(o, datetime.datetime):
#         return o.__str__()
#     if isinstance(o, np.bool_):
#         return o.__str__()

# def writeToJSONFile (filepath, data):
#     with open (filepath,'w') as file:
#         json.dump(data, file, default=myconverter, indent=4)


def prediction(url, independent_variables, dependent_variables=None, path_model=''):
    print("start prediction")
    dataframe = pd.read_csv(url)
    dataframe = dataframe.dropna()
    X_original = pd.DataFrame()
    for var in independent_variables:
        X_original[var] = dataframe[var]

    data = cleaningdata(url, independent_variables, dependent_variables)
    message = data["message"]
    if message == "success":
        X = data["X"]
        original_y = data['y']
        X_std = preprocessing.scale(X)
        sc = StandardScaler()
        sc.fit(X_std)
        X_std = sc.transform(X_std)
        model = path_model  # urllib.request.urlopen(path_model).read()
        # model = jsonpickle.decode(json.loads(model))
        mp, pickled_Xtrain, pickled_Ytrain, pickled_score = pickle.load(open(model, 'rb'))
        print(pickled_Xtrain.shape)
        print(pickled_Ytrain.shape)
        y_Predicted = mp.predict(X_std)
        predicted_data = []
        for (index, row), y in zip(X_original.iterrows(), y_Predicted):
            indep_test = {}
            colonnename = []
            colvalues = []
            for colname in X.columns:
                colonnename.append(colname)
                colvalues.append(row[colname])
                indep_test[colname] = row[colname]
            data = indep_test
            y_name = pickled_Ytrain.columns[0] + "_predicted"
            data[y_name] = y
            predicted_data.append(data)
            # plt.plot(y_Predicted,color='red')
            original_y = np.array(original_y)[:, 0]
            y_Predicted = np.array(y_Predicted)
            plt.scatter(original_y, y_Predicted, c='black', marker='.')
            print(original_y.shape, y_Predicted.shape)
            print(np.corrcoef(original_y, y_Predicted)[0, 1])
            plt.axis('equal')
            abline(1)
            plt.show()

            ratios = original_y / y_Predicted
            # ratios = ratios[np.logical_and(ratios < 10, ratios > 1/10)]
            _, bins = np.histogram(ratios, bins = 20)
            print(bins)
            # plt.show()
            logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            plt.hist(ratios, bins = logbins)
            plt.xscale('log')
            plt.show()
            return
        # writeToJSONFile("predicted_data.json", predicted_data)
    else:
        return message


# "sequences_resumed_last_version.csv"
# "sequences_resumed.csv"
# "period_pir0", "period_pir0", "period_pir1", "period_pir2", "period_pir3", "delta_time_0_2",
#                              "delta_time_1_3", "corridor", "direction"
res, y1, y2 = NeuralNetwork("sequences_resumed.csv", .10,
                            ["time_pir0", "time_pir1", "time_pir2", "time_pir3",
                             "period_pir0", "period_pir0", "period_pir1", "period_pir2",
                             "ampl_pir0", "ampl_pir1", "ampl_pir2", "ampl_pir3",
                             "period_pir3", "delta_time_0_2",
                             "delta_time_1_3", "corridor", "direction"], ["walking_speed"], None, data_multiplier=1)
prediction("sequences_resumed.csv",
           ["time_pir0", "time_pir1", "time_pir2", "time_pir3",
            "period_pir0", "period_pir0", "period_pir1", "period_pir2",
            "ampl_pir0", "ampl_pir1", "ampl_pir2", "ampl_pir3",
            "period_pir3", "delta_time_0_2",
            "delta_time_1_3", "corridor", "direction"], ['walking_speed'], "NeuralNetworkModel")
