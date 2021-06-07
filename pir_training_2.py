import datetime

import dateutil
import pandas as pd
import scipy
import scipy.signal

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neural_network
import sklearn.neighbors
import sklearn.gaussian_process
import sklearn.discriminant_analysis
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes

from matplotlib import pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')


# Split all the data into different sequences (defined by the "sequence_id")
def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(pir_df[pir_df['sequence_id'] == seq_id])
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



#
# # Method 1: Take the features from the biggest peak of each sequence
# number_of_peaks_to_analyse = 1
# # Method 2: Take the features from more peaks (here 3)
# # number_of_peaks = 3
#
# df = pd.read_csv("all_pir_values.csv")
# all_sequences = split_sequences(df)
# X = []
# Y = []
# # Compute the feature vectors for all sequences
# for sequence in all_sequences:
#     features = get_sequence_features(sequence, number_of_peaks_to_analyse)
#     Y.append(features[0])
#     X.append(features[1])
#
# # Check if the the feature vectors all have the same size
# # For some of them, there are not enough peaks in the signal
# feature_vector_length = int(np.median([len(x) for x in X]))
# for i in range(len(X)-1, 0, -1):
#     if len(X[i]) != feature_vector_length: # The sequences with too few peaks are ignored
#         del X[i]
#         del Y[i]





##########################
### Part for direction ###
##########################


# df_sequences = pd.read_csv("sequences_resumed.csv")
# df_sequences = df_sequences.dropna()
# input_cols = ['period_pir0', 'time_pir0', 'period_pir1', 'time_pir1', 'period_pir2', 'time_pir2', 'period_pir3', 'time_pir3', 'delta_time_0_2', 'delta_time_1_3']
# X = df_sequences[input_cols]
# Y = df_sequences['direction']
#
#
# # Split the data in 2 : training/testing
# train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)
#
# # We use most of the regression methods provided by sklearn
# classifiers = [
#     sklearn.neighbors.KNeighborsClassifier(3),
#     sklearn.svm.SVC(kernel="linear", C=0.025),
#     sklearn.svm.SVC(gamma=2, C=1),
#     sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
#     sklearn.tree.DecisionTreeClassifier(max_depth=5),
#     sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),
#     sklearn.ensemble.AdaBoostClassifier(),
#     sklearn.naive_bayes.GaussianNB(),
#     sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
# ]
#
# pd.set_option('float_format', '{:.3f}'.format)
# # Display stats about the velocities of the sequences
# print("Directions:")
# print(pd.DataFrame(Y.values, columns=["Directions"]).describe())
#
# results = []
# # Train all regressors
# # Results are given using Cross-Validation with 5 folds
# for item in classifiers:
#     clf = item
#     clf.fit(train_X, train_y)
#     cv_score_accuracy = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'accuracy')
#     # cv_score_precision = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'precision')
#     cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
#     results.append([item.__class__.__name__, np.mean(cv_score_accuracy), "np.mean(cv_score_precision)", np.mean(cv_score_r2)])
#
# # Display the results
# print(pd.DataFrame(results, columns=["Classifier's name", "Accuracy", "Precision", "R2 score"]))
#
# predictions = classifiers[-1].predict(X)
# ground_truth = Y.values
#
# predictions = [x for _, x in sorted(zip(Y.index, predictions))]
# ground_truth = [x for _, x in sorted(zip(Y.index, ground_truth))]
#
# plt.plot(predictions, label="Prediction")
# plt.plot(ground_truth, label="Ground Truth")
# plt.legend()
# plt.show()


##########################
### Part for corridor ###
##########################


df_sequences = pd.read_csv("sequences_resumed.csv")
df_sequences = df_sequences.dropna()
input_cols = ['period_pir0', 'time_pir0', 'period_pir1', 'time_pir1', 'period_pir2', 'time_pir2', 'period_pir3', 'time_pir3', 'delta_time_0_2', 'delta_time_1_3']
X = df_sequences[input_cols]
Y = df_sequences['corridor']


# Split the data in 2 : training/testing
train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)

# We use most of the regression methods provided by sklearn
classifiers = [
    sklearn.neighbors.KNeighborsClassifier(3),
    sklearn.svm.SVC(kernel="linear", C=0.025),
    sklearn.svm.SVC(gamma=2, C=1),
    sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
    sklearn.tree.DecisionTreeClassifier(max_depth=10),
    sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=10, max_features=10),
    sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),
    sklearn.ensemble.AdaBoostClassifier(),
    sklearn.naive_bayes.GaussianNB(),
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
]

pd.set_option('float_format', '{:.3f}'.format)
# Display stats about the velocities of the sequences
print("Corridor:")
print(pd.DataFrame(Y.values, columns=["Corridor"]).describe())

results = []
# Train all regressors
# Results are given using Cross-Validation with 5 folds
for item in classifiers:
    clf = item
    clf.fit(train_X, train_y)
    cv_score_accuracy = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'accuracy')
    # cv_score_precision = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'precision')
    cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
    results.append([item.__class__.__name__, np.mean(cv_score_accuracy), "np.mean(cv_score_precision)", np.mean(cv_score_r2)])

# Display the results
print(pd.DataFrame(results, columns=["Classifier's name", "Accuracy", "Precision", "R2 score"]))

predictions = classifiers[-1].predict(X)
ground_truth = Y.values

predictions = [x for _, x in sorted(zip(Y.index, predictions))]
ground_truth = [x for _, x in sorted(zip(Y.index, ground_truth))]

plt.plot(predictions, label="Prediction")
plt.plot(ground_truth, label="Ground Truth")
plt.legend()
plt.show()



df_sequences = pd.read_csv("sequences_resumed.csv")
df_sequences = df_sequences.dropna()
input_cols = ['period_pir0', 'time_pir0', 'period_pir1', 'time_pir1', 'period_pir2', 'time_pir2', 'period_pir3', 'time_pir3', 'delta_time_0_2', 'delta_time_1_3']
X = df_sequences[input_cols]
# X['direction'] = classifiers[0].predict(X)
Y = df_sequences['walking_speed']


# Split the data in 2 : training/testing
train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5)

# We use most of the regression methods provided by sklearn
regression_models = [
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
    sklearn.linear_model.HuberRegressor(),

]

pd.set_option('float_format', '{:.3f}'.format)
# Display stats about the velocities of the sequences
print("Speeds recorded:")
print(pd.DataFrame(Y.values, columns=["Speeds recorded on each sequence (km/h)"]).describe())

results = []
# Train all regressors
# Results are given using Cross-Validation with 5 folds
for item in regression_models:
    clf = item
    clf.fit(train_X, train_y)
    cv_score_mean_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_mean_absolute_error')
    cv_score_abs_error = -sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'neg_median_absolute_error')
    cv_score_r2 = sklearn.model_selection.cross_val_score(clf, test_X, test_y, cv=5, scoring = 'r2')
    results.append([item.__class__.__name__, np.mean(cv_score_mean_error), np.mean(cv_score_abs_error), np.mean(cv_score_r2)])

# Display the results
print(pd.DataFrame(results, columns=["Regressor name", "Mean error", "Median abs error", "R2 score"]))

predictions = regression_models[-1].predict(X)
ground_truth = Y.values

predictions = [x for _, x in sorted(zip(Y.index, predictions))]
ground_truth = [x for _, x in sorted(zip(Y.index, ground_truth))]

plt.plot([0, 10], [0, 10], '--', color=(0, 0, 0, .5))
plt.scatter(predictions, ground_truth)
plt.xlim((0, 10))
plt.ylim((0, 10))
# plt.plot(predictions, label="Prediction")
# plt.plot(ground_truth, label="Ground Truth")
# plt.legend()
plt.show()
