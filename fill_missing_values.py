import datetime
import random

import dateutil
import pandas as pd
import scipy
import scipy.signal

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.linear_model
from matplotlib import pyplot as plt

import InfluxDB_connection as db
from warnings import filterwarnings

import training_NN

filterwarnings('ignore')

import improve_speed_values as speed_values

# This is used for sequences where the camera didn't manage to find the user's speed...
# It's very bad usage, but it was useful at first
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

# Get the speed of a sequence which has not been found in the tmp_data JSONs, but for which I'm sure of the ground truth
# This is a solution of last resort
def get_predefined_speed(start, end):
    # Get dates as timestamps
    if isinstance(start, str):
        start = dateutil.parser.isoparse(start.replace('Z', '+00:00')).timestamp() * 1000
    if isinstance(end, str):
        end = dateutil.parser.isoparse(end.replace('Z', '+00:00')).timestamp() * 1000
    if start > end:
        tmp = start
        start = end
        end = tmp

    # Get the corresponding speed and apply some noise to train the Learning machines
    if slow[0] <= start <= slow[1] or slow[0] <= end <= slow[1]:
        return slow_speed + (random.random() * 0.2 + 0.1)
    elif normal1[0] <= start <= normal1[1] or normal2[0] <= start <= normal2[1] or normal1[0] <= end <= normal1[1] or normal2[0] <= end <= normal2[1]:
        return normal_speed + (random.random() * 0.2 + 0.1)
    elif fast1[0] <= start <= fast1[1] or fast2[0] <= start <= fast2[1] or fast1[0] <= end <= fast1[1] or fast2[0] <= end <= fast2[1]:
        return fast_speed + (random.random() * 0.2 + 0.1)


# Get the speed from the tmp_data JSONs created in the recordings
all_speeds = speed_values.get_all_speeds_recorded()
all_precomputed_speeds = speed_values.get_precomputed_speeds()
# Get the speed of a sequence by comparing with the speeds computed in "all_speeds"
def get_my_speed(start, end = None):
    # Get dates as timestamps to compare them
    if isinstance(start, str):
        start = dateutil.parser.isoparse(start.replace('Z', '+00:00')).timestamp() * 1000
    if isinstance(end, str):
        end = dateutil.parser.isoparse(end.replace('Z', '+00:00')).timestamp() * 1000
    if end is not None:
        # Simple check for start < end
        if start > end:
            tmp = start
            start = end
            end = tmp
    else:
        end = start # Looking for 1 unique timestamp

    speeds = []
    # Find the speed corresponding to our sequence
    for i in range(len(all_speeds)):
        if start <= all_speeds[i][0][0] <= end or start <= all_speeds[i][0][1] <= end \
                or all_speeds[i][0][0] <= start <= all_speeds[i][0][1] \
                or all_speeds[i][0][0] <= end <= all_speeds[i][0][1]:
            speeds.append(all_speeds[i][1])
        # speed sequences are ordered, so if the timestamp exceed this value, it will not find anything else
        if end < all_speeds[i][0][0]:
            break
    if len(speeds) > 0:
        # We have multiple subsequences crossing, so we take the median average
        mean_speed = np.median(speeds)
        return mean_speed
    return None

# Get all PIR values from the InfluxDB and associate them with the speed computed
# The reason for this function is that the PIR values and the speed estimation are not
#   joint correctly with InfluxDB, so we need to compute this by ourselves
def join_pir_and_params(all_queries):
    # If two values are separated by 500ms or more, we consider them as 2 different sequences
    delay_limit_for_sequence = 500

    all_data_joint = []
    last_pir_timestamp = None
    last_pir_values = None
    last_walking_speed_timestamp = None
    last_walking_speed = None
    sequence_id = -1

    not_tagged_query = all_queries.items()[0][1]
    # speeds = {dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000: data['walking_speed'] for data in not_tagged_query}
    starting_sequence_time = 0
    offset = 0
    for key, all_data in all_queries.items():
        # We don't take into account sequences for which we didn't give a tag
        if key[1]['move'] == '':
            continue

        for data_pir in all_data:
            offset += 0.01
            current_timestamp = dateutil.parser.isoparse(data_pir['time'].replace('Z', '+00:00')).timestamp() * 1000

            if data_pir['pir0'] is not None:
                last_pir_values = data_pir
                last_pir_timestamp = current_timestamp

            if last_pir_values is not None \
                    and (current_timestamp - last_pir_timestamp) < delay_limit_for_sequence:
                this_data = last_pir_values

                if len(all_data_joint) == 0 or (current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000) > delay_limit_for_sequence:
                    sequence_id += 1
                    starting_sequence_time = current_timestamp

                # speed = 0
                # if slow[0] <= current_timestamp <= slow[1]:
                #     speed = slow_speed
                # elif normal1[0] <= current_timestamp <= normal1[1] or normal2[0] <= current_timestamp <= normal2[1]:
                #     speed = normal_speed
                # elif fast1[0] <= current_timestamp <= fast1[1] or fast2[0] <= current_timestamp <= fast2[1]:
                #     speed = fast_speed

                # Find the precomputed speed from this sequence
                # this_data['walking_speed'] = get_my_speed(starting_sequence_time, current_timestamp)
                # Find the precomputed speed from this timestamp
                this_data['walking_speed'] = get_my_speed(current_timestamp)
                # if this_data['walking_speed'] is None:
                #   this_data['walking_speed'] = speed # * ((noise([offset]) * 0.3) + 0.8)  + (random.random()*0.2+0.1)
                # Assign a sequence ID
                this_data['sequence_id'] = sequence_id
                move = key[1]['move']
                this_data['move'] = move
                this_data['direction'] = 1 if 'NS' in move else -1
                this_data['corridor'] = 1 if move[0] == 'O' else 0 if move[0] == 'C' else -1
                if len(all_data_joint) == 0 or this_data['time'] != all_data_joint[-1]['time']:
                    all_data_joint.append(this_data)

    return all_data_joint

# Creates an array with one dataframe per sequence
def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(pir_df[pir_df['sequence_id'] == seq_id])
    return all_sequences

# Extract features from one PIR sensor of a sequence
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

# Extract features from a sequence
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
    # Access the DB
    client = db.ListicClientDB(server_side=True)
    all_queries = client.get_all_pir_values()
    # Get all the values and get the speed associated
    df = pd.DataFrame(join_pir_and_params(all_queries))

    # For each sequence, spread the speed associated with all the data of the sequence
    for seq_id in df['sequence_id'].unique():
        # Get the median of the sequence just to get the first non-null value
        median = df[df['sequence_id'] == seq_id]['walking_speed'].mean()
        # If it's still null, try to associate the predefined speed
        # if np.isnan(median):
        #     start = df[df['sequence_id'] == seq_id].iloc[0]['time']
        #     end = df[df['sequence_id'] == seq_id].iloc[-1]['time']
        #     median = get_predefined_speed(start, end)

        # Give this speed to all data where we didn't find the correct speed
        df['walking_speed'][np.logical_and(df['sequence_id'] == seq_id, np.isnan(df['walking_speed']))] = median

        # Remove sequences of only one data (it used to happen at first, maybe not anymore, now)
        if len(df[df['sequence_id'] == seq_id]) <= 1:
            df = df[df['sequence_id'] != seq_id]

    # Remove sequences where the speed wasn't found
    df.dropna(inplace=True)
    df = df[np.isfinite(df['walking_speed'])]
    # Create the "compteur" variable in the dataframe
    df['compteur'] = range(len(df.index))
    # Save it!
    df.to_csv("all_pir_values.csv")

    df = pd.read_csv("all_pir_values.csv")


    all_sequences = split_sequences(df)
    X = []
    Y = []
    # Create the X and y data for the training
    for sequence in all_sequences:
        features = get_sequence_features(sequence)
        Y.append(features[0])
        X.append(features[1])

    # Remove sequences for which we weren't able to find enough features (ex: not enough peaks)
    feature_vector_length = max([len(x) for x in X])
    for i in range(len(X)-1, 0, -1):
        if len(X[i]) != feature_vector_length:
            del X[i]
            del Y[i]

    # Split in training and testing sets
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)

    # Use and compare all these regression models
    regressors = [
        sklearn.svm.SVR(),
        sklearn.svm.LinearSVR(),
        # sklearn.linear_model.SGDRegressor(),
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

    # Describe the speed values
    pd.set_option('float_format', '{:.3f}'.format)
    print("Speeds recorded:")
    print(pd.DataFrame(Y, columns=["Speeds recorded on each sequence (km/h)"]).describe())

    results = []
    for item in regressors:
        clf = item
        clf.fit(train_X, train_y)
        # cv_score_percent_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = sklearn.metrics.mean_absolute_percentage_error)
        cv_score_mean_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_mean_absolute_error')
        cv_score_abs_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_median_absolute_error')
        cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
        # # print(item.__class__.__name__, ":", -np.mean(cv_score)) #sklearn.metrics.mean_absolute_error(clf.predict(test_X), test_y))
        results.append([item.__class__.__name__, np.mean(cv_score_mean_error), np.mean(cv_score_abs_error), np.mean(cv_score_r2)])

    print(pd.DataFrame(results, columns=["Regressor name", "Mean error", "Median abs error", "R2 score"]))

    import subprocess
    subprocess.call(['/usr/bin/Rscript', '--vanilla', '/home/marchartley/Downloads/Test1.R'])

    res_without_dir = training_NN.NeuralNetwork("sequences_resumed.csv", 0.3,
                                    ["period_pir0", "period_pir1", "period_pir2", "period_pir3", "delta_time_0_2",
                                     "delta_time_1_3"],
                                    ["walking_speed"], None)

    df_res_without_dir = pd.DataFrame(res_without_dir.cv_results_)
    # df_res_with_dir = pd.DataFrame(res_with_dir.cv_results_)

    df_res_without_dir.to_csv("res_without_dir_speeds_under_2_0.csv")
    # df_res_with_dir.to_csv("res_with_dir_speeds_under_2_0.csv")

    plt.show()
