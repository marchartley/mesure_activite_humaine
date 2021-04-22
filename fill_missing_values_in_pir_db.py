import datetime

import dateutil
import pandas as pd
import scipy
import scipy.signal

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model

import InfluxDB_connection as db


def join_pir_and_params(all_queries):
    all_data_joint = []
    last_pir_timestamp = None
    last_pir_values = None
    last_walking_speed_timestamp = None
    last_walking_speed = None
    sequence_id = -1
    for all_data in all_queries:
        for data in all_data:
            current_timestamp = dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000
            if data['pir0'] is not None:
                last_pir_values = data
                last_pir_timestamp = current_timestamp
            if data['walking_speed'] is not None:
                last_walking_speed_timestamp = current_timestamp
                if last_walking_speed is None:
                    last_walking_speed = data['walking_speed']
                else:
                    last_walking_speed = (data['walking_speed'] + last_walking_speed)/2

            if last_pir_values is not None and last_walking_speed is not None and (current_timestamp - last_walking_speed_timestamp) < 1000:
                this_data = last_pir_values
                this_data['walking_speed'] = last_walking_speed
                if len(all_data_joint) == 0 or (current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000) > 100:
                    # if len(all_data_joint) > 0:
                    #     print(f"{sequence_id} -> {sequence_id+1} : {(current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000)}")
                    sequence_id += 1
                this_data['sequence_id'] = sequence_id
                if len(all_data_joint) == 0 or this_data['time'] != all_data_joint[-1]['time']:
                    all_data_joint.append(this_data)
    return all_data_joint

def split_sequences(pir_df:pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(df[df['sequence_id'] == seq_id])
    return all_sequences

def get_signal_features(signal, number_of_peaks = 5):
    peaks, props = scipy.signal.find_peaks(signal, height = 0)
    # Sort by prominence
    peaks = sorted(peaks, key = lambda peak: scipy.signal.peak_prominences(signal, [peak])[0], reverse = True)
    peaks = peaks[:number_of_peaks] # Get only th most interesting

    prominences = scipy.signal.peak_prominences(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    widths = scipy.signal.peak_widths(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    first_peak_offset = min(peaks)/len(signal) if len(peaks) == number_of_peaks else 0
    last_peak_offset = max(peaks)/len(signal) if len(peaks) == number_of_peaks else 0

    return [*prominences, *widths, first_peak_offset, last_peak_offset]

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
    speed = np.mean(sequence['walking_speed'])
    # print(props0);exit()
    return speed * 3.6, [*pir0, *pir1, *pir2, *pir3]

# client = db.ListicClientDB(server_side=True)
# all_queries = client.get_all_pir_values()
# df = pd.DataFrame(join_pir_and_params(all_queries))
# for i in range(df['sequence_id'].max() + 1):
#     if len(df[df['sequence_id'] == i]) <= 1:
#         df = df[df['sequence_id'] != i]
# print(df)
# df.to_csv("all_pir_values.csv")
# for i in range(30):
#     print(join_pir_and_params(all_queries)[i])

df = pd.read_csv("all_pir_values.csv")
all_sequences = split_sequences(df)
X = []
Y = []
for sequence in all_sequences:
    features = get_sequence_features(sequence)
    Y.append(features[0])
    X.append(features[1])

train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)
# model = sklearn.linear_model.LinearRegression()
# model = sklearn.svm.SVC()
# model.fit(train_X, train_y)

classifiers = [
    sklearn.svm.SVR(),
    sklearn.linear_model.SGDRegressor(),
    sklearn.linear_model.BayesianRidge(),
    sklearn.linear_model.LassoLars(),
    sklearn.linear_model.ARDRegression(),
    sklearn.linear_model.PassiveAggressiveRegressor(),
    sklearn.linear_model.TheilSenRegressor(),
    sklearn.linear_model.LinearRegression()]

for item in classifiers:
    clf = item
    clf.fit(train_X, train_y)
    cv_score = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_median_absolute_error')
    print(item , ":", -np.mean(cv_score)) #sklearn.metrics.mean_absolute_error(clf.predict(test_X), test_y))
