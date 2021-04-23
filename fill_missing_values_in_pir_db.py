import datetime
import random

import dateutil
import pandas as pd
import scipy
import scipy.signal

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model

import InfluxDB_connection as db
from warnings import filterwarnings
filterwarnings('ignore')

import improve_speed_values as speed_values

from perlin_noise import PerlinNoise

noise = PerlinNoise(octaves=10, seed=1)
slow = [dateutil.parser.isoparse('2021-04-21 14:51:30+02:00').timestamp() * 1000, dateutil.parser.isoparse('2021-04-21 15:17:10+02:00').timestamp() * 1000]
normal1 = [dateutil.parser.isoparse('2021-04-21 15:17:10+02:00').timestamp() * 1000, dateutil.parser.isoparse('2021-04-21 15:42:54+02:00').timestamp() * 1000]
normal2 = [dateutil.parser.isoparse('2021-04-21 15:58:50+02:00').timestamp() * 1000, dateutil.parser.isoparse('2021-04-21 16:05:10+02:00').timestamp() * 1000]
fast1 = [dateutil.parser.isoparse('2021-04-21 15:42:54+02:00').timestamp() * 1000, dateutil.parser.isoparse('2021-04-21 15:48:00+02:00').timestamp() * 1000]
fast2 = [dateutil.parser.isoparse('2021-04-21 16:05:10+02:00').timestamp() * 1000, dateutil.parser.isoparse('2021-04-21 16:14:47+02:00').timestamp() * 1000]

slow_speed = 0.70
normal_speed = 1.1
fast_speed = 2.3


# 1619010068159.0 1619009944864.0


def get_my_speed(start, end):
    all_speeds = speed_values.get_all_speeds_recorded()
    if start > end:
        tmp = start
        start = end
        end = tmp
    for i in range(len(all_speeds)):
        if start <= all_speeds[i][0][0] <= end or start <= all_speeds[i][0][1] <= end \
                or all_speeds[i][0][0] <= start <= all_speeds[i][0][1] \
                or all_speeds[i][0][0] <= end <= all_speeds[i][0][1]:
            return all_speeds[i][1]
    else:
        print("Nothing found")
    return None

def join_pir_and_params(all_queries):
    delay_limit_for_sequence = 1000

    all_data_joint = []
    last_pir_timestamp = None
    last_pir_values = None
    last_walking_speed_timestamp = None
    last_walking_speed = None
    sequence_id = -1

    not_tagged_query = all_queries.items()[0][1]
    speeds = {dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000: data['walking_speed'] for data in not_tagged_query}
    # print(speeds)
    # exit()
    starting_sequence_time = 0
    offset = 0
    num_true = 0
    num_ttl = 0
    for key, all_data in all_queries.items():
        if key[1]['move'] == '':
            continue
        for data_pir in all_data:
            offset += 0.01
            current_timestamp = dateutil.parser.isoparse(data_pir['time'].replace('Z', '+00:00')).timestamp() * 1000

            if data_pir['pir0'] is not None:
                last_pir_values = data_pir
                last_pir_timestamp = current_timestamp
            if current_timestamp in speeds is not None:
                last_walking_speed_timestamp = current_timestamp
                if last_walking_speed is None:
                    last_walking_speed = speeds[current_timestamp]
                else:
                    last_walking_speed = (speeds[current_timestamp] + last_walking_speed)/2

            if last_pir_values is not None and last_walking_speed is not None and (current_timestamp - last_walking_speed_timestamp) < delay_limit_for_sequence:
                this_data = last_pir_values

                if len(all_data_joint) == 0 or (current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000) > delay_limit_for_sequence:
                    # if len(all_data_joint) > 0:
                    #     print(f"{sequence_id} -> {sequence_id+1} : {(current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000)}")
                    sequence_id += 1
                    starting_sequence_time = current_timestamp

                speed = 0
                if slow[0] <= current_timestamp <= slow[1]:
                    speed = slow_speed
                elif normal1[0] <= current_timestamp <= normal1[1] or normal2[0] <= current_timestamp <= normal2[1]:
                    speed = normal_speed
                elif fast1[0] <= current_timestamp <= fast1[1] or fast2[0] <= current_timestamp <= fast2[1]:
                    speed = fast_speed

                # this_data['walking_speed'] = get_my_speed(starting_sequence_time, current_timestamp)
                # if this_data['walking_speed'] is None:
                this_data['walking_speed'] = ((noise([offset]) * 0.3) + 0.8) * speed + (random.random()*0.2+0.1)
                # this_data['walking_speed'] = last_walking_speed
                this_data['sequence_id'] = sequence_id
                move = key[1]['move']
                this_data['move'] = move
                this_data['direction'] = 1 if 'NS' in move else -1
                this_data['corridor'] = 1 if move[0] == 'O' else 0 if move[0] == 'C' else -1
                if len(all_data_joint) == 0 or this_data['time'] != all_data_joint[-1]['time']:
                    all_data_joint.append(this_data)

        current_sequence = -1
        start_sequence = 0
        for i in range(len(all_data_joint)):
            num_ttl += 1
            if current_sequence != all_data_joint[i]['sequence_id']:
                start_sequence = dateutil.parser.isoparse(all_data_joint[i]['time'].replace('Z', '+00:00')).timestamp() * 1000
                current_sequence = all_data_joint[i]['sequence_id']
                speed_found = None
                for j in range(len(all_data_joint) - i):
                    if all_data_joint[j]['sequence_id'] != current_sequence:
                        speed_found = get_my_speed(start_sequence, dateutil.parser.isoparse(all_data_joint[j]['time'].replace('Z', '+00:00')).timestamp() * 1000)
                        if speed_found is not None:
                            for x in range(i, j):
                                all_data_joint[x]['walking_speed'] = speed_found
                                num_true += 1
                            break
                        break
    print((num_true/num_ttl)*100)
    return all_data_joint

def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(pir_df[pir_df['sequence_id'] == seq_id])
    return all_sequences

def get_signal_features(signal, number_of_peaks = 3):
    signal = np.array(signal)
    peaks, props = scipy.signal.find_peaks(signal, height = 0)
    # Sort by prominence
    peaks = sorted(peaks, key = lambda peak: scipy.signal.peak_prominences(signal, [peak])[0], reverse = True)
    peaks = peaks[:number_of_peaks] # Get only th most interesting

    prominences = scipy.signal.peak_prominences(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    widths = scipy.signal.peak_widths(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    peaks_offsets = [peak/len(signal) for peak in peaks]
    # first_peak_offset = min(peaks)/len(signal) if len(peaks) == number_of_peaks else 0
    # last_peak_offset = max(peaks)/len(signal) if len(peaks) == number_of_peaks else 0

    return [*prominences, *widths, *peaks_offsets] # first_peak_offset, last_peak_offset]

    # number_of_peaks = len(peaks)
    # max_peak_height = max(props['peak_heights']) if number_of_peaks > 0 else 0
    # prominence = np.mean(scipy.signal.peak_prominences(signal, peaks)) if number_of_peaks > 0 else 0
    # widths = np.mean(scipy.signal.peak_widths(signal, peaks)) if number_of_peaks > 0 else 0
    # first_peak_offset = peaks[0]/len(signal) if number_of_peaks > 0 else -1
    # last_peak_offset = peaks[-1]/len(signal) if number_of_peaks > 0 else -1
    # return [number_of_peaks, max_peak_height, prominence, widths, first_peak_offset, last_peak_offset]

def get_sequence_features(sequence):
    pir0 = get_signal_features(sequence['pir0'])
    pir1 = get_signal_features(sequence['pir1'])
    pir2 = get_signal_features(sequence['pir2'])
    pir3 = get_signal_features(sequence['pir3'])
    direction = int(sequence['direction'].iloc[0])
    corridor = int(sequence['corridor'].iloc[0])
    speed = np.median(sequence['walking_speed'])

    speed_as_km_per_hour = speed * 3.6 # * direction
    return speed_as_km_per_hour, [direction, corridor, *pir0, *pir1, *pir2, *pir3]



if __name__ == "__main__":
    client = db.ListicClientDB(server_side=True)
    all_queries = client.get_all_pir_values()
    df = pd.DataFrame(join_pir_and_params(all_queries))
    for seq_id in df['sequence_id'].unique():
        if len(df[df['sequence_id'] == seq_id]) <= 1:
            df = df[df['sequence_id'] != seq_id]
    # # print(df[df['sequence_id'] == 1])
    # # exit()
    df.to_csv("all_pir_values.csv")

    df = pd.read_csv("all_pir_values.csv")
    all_sequences = split_sequences(df)
    X = []
    Y = []
    for sequence in all_sequences:
        features = get_sequence_features(sequence)
        Y.append(features[0])
        X.append(features[1])

    feature_vector_length = int(np.median([len(x) for x in X]))
    for i in range(len(X)-1, 0, -1):
        if len(X[i]) != feature_vector_length:
            del X[i]
            del Y[i]

    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)

    classifiers = [
        sklearn.svm.SVR(),
        sklearn.svm.LinearSVR(),
        sklearn.linear_model.SGDRegressor(),
        sklearn.linear_model.BayesianRidge(),
        sklearn.linear_model.LassoLars(),
        sklearn.linear_model.ARDRegression(),
        sklearn.linear_model.PassiveAggressiveRegressor(),
        sklearn.linear_model.TheilSenRegressor(),
        sklearn.linear_model.LinearRegression(),
        sklearn.linear_model.Ridge(),
        sklearn.linear_model.ElasticNet(),
        # sklearn.linear_model.LogisticRegression(),
        # sklearn.linear_model.TweedieRegressor(),
        # sklearn.linear_model.Perceptron(),
        sklearn.linear_model.HuberRegressor(),

    ]

    pd.set_option('float_format', '{:.3f}'.format)
    print("Speeds recorded:")
    print(pd.DataFrame(Y, columns=["Speeds recorded on each sequence (km/h)"]).describe())

    results = []
    for item in classifiers:
        clf = item
        clf.fit(train_X, train_y)
        # cv_score_percent_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = sklearn.metrics.mean_absolute_percentage_error)
        cv_score_mean_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_mean_absolute_error')
        cv_score_abs_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_median_absolute_error')
        cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
        # # print(item.__class__.__name__, ":", -np.mean(cv_score)) #sklearn.metrics.mean_absolute_error(clf.predict(test_X), test_y))
        results.append([item.__class__.__name__, np.mean(cv_score_mean_error), np.mean(cv_score_abs_error), np.mean(cv_score_r2)])

    print(pd.DataFrame(results, columns=["Regressor name", "Mean error", "Median abs error", "R2 score"]))
    #
    # i = 0
    # for x_test, y_test in zip(test_X, test_y):
    #     print(classifiers[0].predict([x_test]), y_test)
    #     i += 1
    #     if i > 10:
    #         break
    #
    # print('weights: ')
    # print(classifiers[0].coef_)
    # print('Intercept: ')
    # print(classifiers[0].intercept_)
