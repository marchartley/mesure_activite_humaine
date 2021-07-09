import math
import random

import sklearn.model_selection
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
import time
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

import shap

def abline(slope, intercept=0):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.concatenate([np.array(axes.get_xlim()), np.array(axes.get_ylim())])
    x_vals.sort()
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def cleaningdata(url, independent_variables, dependent_variable):
    df = pd.read_csv(url).dropna()
    df = df[df['walking_speed'] < 2.0]
    # df = df[df['corridor'] == 1]
    df['freq_pir0'] = 1 / df['period_pir0']
    df['freq_pir1'] = 1 / df['period_pir1']
    df['freq_pir2'] = 1 / df['period_pir2']
    df['freq_pir3'] = 1 / df['period_pir3']

    return {
        'message': 'success',
        'X': df[independent_variables],
        'y': df[dependent_variable] if dependent_variable is not None else []
    }


def splitting_data(url, testsize, independent_variables, dependent_variable):
    data = cleaningdata(url, independent_variables, dependent_variable)
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


def NeuralNetwork(url, testsize, independent_variables, dependent_variable, parameters, model_path):
    response = splitting_data(url, testsize, independent_variables, dependent_variable)
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
                max_iter=500000,
                hidden_layer_sizes=(60, 60, 60, 60),
                # hips
                # speed, direction = (200, 200, 200),
                # corridor_position = (10, 10, 10, 10, 10, 10, 10),
                # cycle_time, cadence = (200, ),
                # dbl_support = (60, 60, 60, 60, 60, 60, 60),
                # stride length = (200, 200),

                # ankles
                # speed, direction = (200, 200, 200),
                # corridor_position = (100, 100, 100),
                # cycle_time = (60, 60, 60),
                # cadence = (200, 200, 200, 200),
                # dbl_support = (60, 60, 60, 60),
                # stride length = (60, 60, 60, 60),
                activation="relu",
                random_state=50,
                alpha=0.1,
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.0005,
                shuffle=True,
            )
            # regressor = MultiOutputRegressor(regressor)
            regressor = RegressorChain(regressor)

        time1 = time.time()
        regressor.fit(X_train_std, y_train)

        time2 = time.time()
        # plt.title("Loss curve")
        # plt.xlabel("nb epochs")
        # plt.ylabel("Loss")
        # plt.plot(regressor.loss_curve_)
        # plt.show()
        print("duration of fitting the model:", (time2 - time1) / 60)
        y_predict_all = regressor.predict(X_test_std)

        for i in range(y_predict_all.shape[1]):
            print(dependent_variables[i])
            y_actual = y_test.values[:, i]
            y_predict = y_predict_all[:, i]
            r_squared_score = r2_score(y_actual, y_predict)
            mse = mean_squared_error(y_actual, y_predict)
            rmse = mean_squared_error(y_actual, y_predict, squared=False
            )
            mae = mean_absolute_error(y_actual, y_predict)
            variance = explained_variance_score(y_actual, y_predict)
            # maximum_error = max_error(y_actual, y_predict)
            mediane = median_absolute_error(y_actual, y_predict)
            percent_error = abs((y_actual - y_predict) / y_actual).mean() * 100
            try:
                mpd = mean_poisson_deviance(y_actual, y_predict)
                mgd = mean_gamma_deviance(y_actual, y_predict)
            except Exception as e:
                mpd = None
                mgd = None
            tuple_objects = (regressor, X_train, y_train, rmse)
            print("variance", variance)
            print("median", mediane)
            print("mse", mse)
            print("rmse", rmse)
            # print("max error", maximum_error)
            print("percent error", percent_error)
            print()

            pickle.dump(tuple_objects, open(model_path, "wb"))
            resultBytes = pickle.dumps(tuple_objects)

        Y_ac = (y_actual)
        Y_pr = (y_predict)

        plt.scatter(y_test.values[:, 0], y_predict_all[:, 0], c=(1, 0, 0, 1), label=dependent_variable[0])
        plt.scatter(y_test.values[:, 1], y_predict_all[:, 1], c=(0, 1, 0, 1), label=dependent_variable[1])
        plt.scatter(y_test.values[:, 2], y_predict_all[:, 2], c=(0, 0, 1, 1), label=dependent_variable[2])
        abline(1)
        plt.show()
        return resultBytes, Y_ac, Y_pr


def prediction(url, independent_variables, dependent_variables=None, path_model='', test_size = 0.3):

    data = splitting_data(url, test_size, independent_variables, dependent_variables)
    message = data["message"]
    if message == "success":
        X = data["X_test"]
        original_y = data['y_test']

        X_std = preprocessing.scale(X)
        sc = StandardScaler()
        sc.fit(X_std)
        X_std = sc.transform(X_std)
        model = path_model

        # X_std = X_std[original_y[dependent_variables[0]] < 1.6]
        # original_y = original_y[original_y[dependent_variables[0]] < 1.6]

        mp, pickled_Xtrain, pickled_Ytrain, pickled_score = pickle.load(open(model, 'rb'))

        # explainer = shap.KernelExplainer(mp.predict, X_std)
        # shap.summary_plot(explainer.shap_values(X_std, nsamples = 50), X_std, feature_names=[
        #     var.replace('_', ' ').capitalize() for var in independent_variables])

        y_Predicted = mp.predict(X_std)

        y_Predicted = y_Predicted
        y_Truth = np.array(original_y)

        # df = pd.concat([pd.Series(np.ravel(y_Truth), name = 'real'), pd.Series(y_Predicted, name = 'prediction')], axis = 1)

        stats = {
            'mse': mean_squared_error(y_Truth, y_Predicted),
            'rmse': mean_squared_error(
                y_Truth, y_Predicted, squared=False
            ),
            'mae': mean_absolute_error(y_Truth, y_Predicted),
            'variance': explained_variance_score(y_Truth, y_Predicted),
            # 'maximum_error': max_error(y_Truth, y_Predicted),
            'mediane': median_absolute_error(y_Truth, y_Predicted),
            'percent_error': abs((y_Truth - y_Predicted) / y_Truth).mean() * 100,
        }

        print(stats)
        # df.to_csv(f"MLP_results_{sensors_height}_{dependent_variables[0]}.csv")
        y_Truth = np.array(y_Truth) # [:, 0]
        y_Predicted = np.array(y_Predicted)

        plt.scatter(y_Truth[:, 0], y_Predicted[:, 0], c='red', marker='.', label=dependent_variables[0])
        plt.scatter(y_Truth[:, 1], y_Predicted[:, 1], c='green', marker='.', label=dependent_variables[1])
        plt.scatter(y_Truth[:, 2], y_Predicted[:, 2], c='blue', marker='.', label=dependent_variables[2])
        # pearson_coef, p_value = pearsonr(y_Truth, y_Predicted)
        # plt.title(f"Correlation : {round(pearson_coef, 2)} ({'p < 0.001' if p_value < 0.001 else 'p = ' + str(round(p_value, 4))})")
        plt.xlabel("Gait speed from depth camera (m/s)")
        plt.ylabel("Gait speed from PIR (m/s)")
        plt.axis('equal')
        abline(1)

        # errors = pd.Series(y_Truth - y_Predicted, name = 'uncertainty (m/s)')
        # abs_errors = pd.Series(abs(y_Truth - y_Predicted), name = 'abs Error (m/s)')
        # ratios = pd.Series(y_Truth / y_Predicted, name = 'Error (%)')
        # ratios[ratios.values < 1.0] = 1 / ratios[ratios.values < 1.0]
        # print(pd.concat([pd.Series((ratios.values - 1) * 100, name = 'Error (%)'), abs_errors, errors], axis=1).describe())
        # plt.show()
        #
        # # plt.hist(errors, bins = 20)
        # errors.plot(kind='hist', density=1)
        # errors.plot(kind='kde')
        # plt.title("Error on the data")
        # plt.xlabel("Prediction error")
        plt.show()

    else:
        return message


sensors_height = 'ankles'
# other sensors heights
# hips, ankles

independent_variables = [
    "time_pir0", "time_pir1", "time_pir2", "time_pir3",
    "ampl_pir0", "ampl_pir1", "ampl_pir2", "ampl_pir3",
    'freq_pir0', 'freq_pir1', 'freq_pir2', 'freq_pir3',
    # 'period_pir0', 'period_pir1', 'period_pir2', 'period_pir3',
    # "delta_time_0_2", "delta_time_1_3",
    # "corridor", "direction"
]
dependent_variables = ['walking_speed', 'stride_length', 'footstep_width']
# Other possible dependent variables
# walking_speed, direction, corridor_position, cycle_time, cadence, double_support_ratio, stride_length, footstep_width

model_path = f"MTLModel_{sensors_height}_{dependent_variables[0]}"
test_size = 0.3
res, y1, y2 = NeuralNetwork(f"sequences_resumed_{sensors_height}.csv", test_size, independent_variables, dependent_variables,
                            None, model_path=model_path)
prediction(f"sequences_resumed_{sensors_height}.csv", independent_variables, dependent_variables, model_path, test_size = test_size)
