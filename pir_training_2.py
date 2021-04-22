import datetime

import dateutil
import pandas as pd
import scipy
import scipy.signal

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model

from warnings import filterwarnings
filterwarnings('ignore')


# Split all the data into different sequences (defined by the "sequence_id")
def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(df[df['sequence_id'] == seq_id])
    return all_sequences

def get_signal_features(signal, number_of_peaks):
    signal = np.array(signal)
    peaks, props = scipy.signal.find_peaks(signal, height = 0)
    # Sort by prominence
    peaks = sorted(peaks, key = lambda peak: scipy.signal.peak_prominences(signal, [peak])[0], reverse = True)

    peaks = peaks[:number_of_peaks]  # Get only th most interesting
    # Get the prominences
    prominences = scipy.signal.peak_prominences(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    # Get the widths
    widths = scipy.signal.peak_widths(signal, peaks)[0] if len(peaks) == number_of_peaks else [0] * number_of_peaks
    # Get the position of the peak in the signal
    peaks_offsets = [peak/len(signal) for peak in peaks]
    return [*prominences, *widths, *peaks_offsets]

def get_sequence_features(sequence, number_of_peaks):
    pir0 = get_signal_features(sequence['pir0'], number_of_peaks=number_of_peaks)
    pir1 = get_signal_features(sequence['pir1'], number_of_peaks=number_of_peaks)
    pir2 = get_signal_features(sequence['pir2'], number_of_peaks=number_of_peaks)
    pir3 = get_signal_features(sequence['pir3'], number_of_peaks=number_of_peaks)
    speed = np.median(sequence['walking_speed'])

    speed_as_km_per_hour = speed * 3.6 # * direction
    # We return the real speed and the feature vector
    # The feature vector has the features from each signal (pi0, pir1, pir2, pir3)
    # The '*' transform the 4 signals into a single flat array
    return speed_as_km_per_hour, [*pir0, *pir1, *pir2, *pir3]

    # direction = int(sequence['direction'].iloc[0])
    # corridor = int(sequence['corridor'].iloc[0])
    # Feature vector is composed of the direction and the corridor we walked in
    # return speed_as_km_per_hour, [direction, corridor, *pir0, *pir1, *pir2, *pir3]




# Method 1: Take the features from the biggest peak of each sequence
number_of_peaks_to_analyse = 1
# Method 2: Take the features from more peaks (here 3)
# number_of_peaks = 3

df = pd.read_csv("all_pir_values.csv")
all_sequences = split_sequences(df)
X = []
Y = []
# Compute the feature vectors for all sequences
for sequence in all_sequences:
    features = get_sequence_features(sequence, number_of_peaks_to_analyse)
    Y.append(features[0])
    X.append(features[1])

# Check if the the feature vectors all have the same size
# For some of them, there are not enough peaks in the signal
feature_vector_length = int(np.median([len(x) for x in X]))
for i in range(len(X)-1, 0, -1):
    if len(X[i]) != feature_vector_length: # The sequences with too few peaks are ignored
        del X[i]
        del Y[i]

# Split the data in 2 : training/testing
train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)

# We use most of the regression methods provided by sklearn
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
    sklearn.linear_model.HuberRegressor(),

]

pd.set_option('float_format', '{:.3f}'.format)
# Display stats about the velocities of the sequences
print("Speeds recorded:")
print(pd.DataFrame(Y, columns=["Speeds recorded on each sequence (km/h)"]).describe())

results = []
# Train all regressors
# Results are given using Cross-Validation with 5 folds
for item in classifiers:
    clf = item
    clf.fit(train_X, train_y)
    cv_score_mean_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_mean_absolute_error')
    cv_score_abs_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_median_absolute_error')
    cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
    results.append([item.__class__.__name__, np.mean(cv_score_mean_error), np.mean(cv_score_abs_error), np.mean(cv_score_r2)])

# Display the results
print(pd.DataFrame(results, columns=["Regressor name", "Mean error", "Median abs error", "R2 score"]))