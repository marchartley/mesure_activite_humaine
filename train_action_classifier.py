import glob
import json
import os
import sys
from collections import deque
from itertools import chain

import pandas as pd
import sklearn
import sklearn.multiclass
import sklearn.svm
import sklearn.cluster
import sklearn.naive_bayes
import sklearn.linear_model

import numpy as np

import InfluxDB_connection
import human_skeleton


def from_video_annotation_to_action_list(filename:str, db_client:InfluxDB_connection.ListicClientDB):
	skeletons2D_and_3D = db_client.retrieve_skeleton_keypoints(start_time = None, end_time = None) # Get ALL keypoints
	# This can be very heavy, we might need to query the DB for each video file instead of all at once
	# For now (01/04/2021) it only represents 2Mo, so it's ok
	json_file = json.load(open(filename, "r"))

	actions_sequences = {}
	for video_filename in json_file:
		for action in json_file[video_filename]:
			if action not in actions_sequences:
				actions_sequences[action] = []
			for video_sequence in json_file[video_filename][action]:
				start = video_sequence['timestamp_start'] / 1000
				end = video_sequence['timestamp_end'] / 1000
				person_id = video_sequence['person_id']
				keypoint_sequence = db_client.retrieve_skeleton_keypoints(start_time = start, end_time = end, skeleton_id = person_id)
				keypoint_sequence = db_client.points_as_json(keypoint_sequence) # Make it serializable
				keypoint_sequence = keypoint_sequence[1] # Take only the 3D part
				keypoint_sequence = keypoint_sequence[person_id] # Don't bother with the "skeleton_id" part
				actions_sequences[action].append(keypoint_sequence)
	return actions_sequences


action_classification_svm = sklearn.multiclass.OneVsOneClassifier(sklearn.svm.LinearSVC(max_iter = 10000))
def process_classification(data, truth, number_of_poses = 10, make_prediction = False):
	global action_classification_svm
	X = []
	Y = []
	for i, frame in enumerate(data):
		kmeans = sklearn.cluster.KMeans(n_clusters = number_of_poses)
		if len(frame) < number_of_poses:
			continue
		predictions = kmeans.fit_predict(frame)

		# Remove predictions duplicates
		sorted_predictions = []
		for prediction in predictions:
			if prediction not in sorted_predictions:
				sorted_predictions.append(prediction)

		# Transform into Kmeans centers
		predict_centers = [kmeans.cluster_centers_[pred] for pred in sorted_predictions]
		# Fit the svm with it
		predict_centers = np.array([predict_centers]).flatten()
		X.append(predict_centers)
		Y.append(truth[i])

	if make_prediction:
		return {'accuracy': action_classification_svm.score(X, Y), 'prediction': action_classification_svm.predict(X)}
	else:
		return action_classification_svm.fit(X, Y)



def calculate_and_normalize_joints_distances(keypoints):
	hips_id = 0
	chest_id = 2

	hips_to_chest_dist = np.linalg.norm(np.array(keypoints[hips_id]) - np.array(keypoints[chest_id]))
	distance_features = []
	for keypoint in keypoints:
		distance_to_hip = np.linalg.norm(np.array(keypoint) - np.array(keypoints[hips_id]))
		distance_features.append(distance_to_hip / hips_to_chest_dist)
	return np.array(distance_features)

def process_training_file_UTKinect_dataset(path_to_training:str):
	content = open(path_to_training, "r+").readlines()
	sequences = {}
	current_sequence_name = ""
	for line in content:
		splitted_line = list(filter(lambda elem: len(elem) > 0, line.strip().split(" ")))
		if len(splitted_line) == 1: # Name of the sequence
			current_sequence_name = splitted_line[0]
			sequences[current_sequence_name] = {}
		else: # Boundaries of an action
			try:
				action_name = splitted_line[0].strip(":")
				start_frame = int(splitted_line[1])
				end_frame = int(splitted_line[2])
				if action_name not in sequences[current_sequence_name]:
					sequences[current_sequence_name][action_name] = []
				sequences[current_sequence_name][action_name].append([start_frame, end_frame])
			except:
				pass

	return sequences

def process_UTKinect_datset(path_to_sequences: str, path_to_training: str):
	sequences = glob.glob(os.path.join(path_to_sequences, "joints*.txt"))

	training_data = process_training_file_UTKinect_dataset(path_to_training)
	actions = []
	X = []
	Y = []

	for sequence in training_data:
		actions += list(training_data[sequence].keys())

	all_actions = list(set(actions)) # Remove duplicates
	current_sequence_id = -1
	skip_frame = False
	for training_sequence_filename in sequences:
		training_sequence_id = os.path.basename(training_sequence_filename).strip(".txt").replace("joints_", "")
		content = open(training_sequence_filename, "r+").readlines()
		for line in content:
			splitted_line = list(filter(lambda elem: elem, line.strip().split(" ")))
			frame_num = int(splitted_line[0])

			current_action = None
			for action in training_data[training_sequence_id]: # Check if the frame is interesting for any action
				for temporal_boundaries in training_data[training_sequence_id][action]:
					if temporal_boundaries[0] <= frame_num <= temporal_boundaries[1]:
						current_action = action
						break
			if current_action is None: # If nothing comes up, ignore this frame
				skip_frame = True
				continue

			# If the previous frames where skipped (no action associated), then increase the ID
			if skip_frame:
				skip_frame = False
				current_sequence_id += 1

			keypoints = []
			for i_keypoint in range((len(splitted_line) - 1) // 3):
				keypoints.append([float(splitted_line[1 + 3 * i_keypoint]),
				                  float(splitted_line[2 + 3 * i_keypoint]),
				                  float(splitted_line[3 + 3 * i_keypoint])])

			if len(X) == current_sequence_id: # First frame of an interesting sequence
				X.append([calculate_and_normalize_joints_distances(keypoints)]) # Append the keypoints of this frame
				Y.append(all_actions.index(current_action)) # And add the action label to the ground truth
			else: # Or add to the list already existing
				X[current_sequence_id].append(calculate_and_normalize_joints_distances(keypoints))

	return X, Y



def process_Florence3D_datset(path_to_training: str):
	content = open(path_to_training, "r+").readlines()

	X = []
	Y = []

	all_actions = list(range(1, 10)) # 1 to 9
	actions_labels = ['', 'raiseArm', 'drinkBottle', 'pickupPhone', 'joinHands', 'kneelForLaces', 'sitDown', 'standUp', 'lookWatch', 'bending']

	current_video = -1
	current_actor = -1
	current_action = -1

	current_sequence_id = -1
	for line in content:
		splitted_line = line.strip().split(" ")
		id_video = int(splitted_line[0])
		id_actor = int(splitted_line[1])
		id_action = int(splitted_line[2])
		if current_video != id_video or current_actor != id_actor or current_action != id_action:
			current_sequence_id += 1
			current_video = id_video
			current_actor = id_actor
			current_action = id_action

		keypoints = []
		for i_keypoint in range((len(splitted_line) - 3) // 3):
			keypoints.append([float(splitted_line[3 + 3 * i_keypoint]),
			                  float(splitted_line[4 + 3 * i_keypoint]),
			                  float(splitted_line[5 + 3 * i_keypoint])])

		if len(X) == current_sequence_id: # First frame of an interesting sequence
			X.append([calculate_and_normalize_joints_distances(keypoints)]) # Append the keypoints of this frame
			Y.append(id_action) # And add the action label to the ground truth
		else: # Or add to the list already existing
			X[current_sequence_id].append(calculate_and_normalize_joints_distances(keypoints))

	return X, Y

def process_KARD_datset(path_to_videos: str):
	video_filenames = glob.glob(path_to_videos + "/*.mp4")

	X = []
	Y = []

	all_actions = list(range(1, 19)) # 1 to 18
	actions_labels = ['', 'Horizontal arm wave', 'High arm wave', 'Two hands wave',
	                  'Catch cap', 'High throw', 'Draw X', 'Draw tick', 'Toss paper',
	                  'Forward kick', 'Side kick', 'Take umbrella', 'Bend', 'Clap hands',
	                  'Walk', 'Phone call', 'Drink', 'Sit down', 'Stand up']

	current_video = -1
	current_actor = -1
	current_action = -1

	current_sequence_id = -1
	for filename in video_filenames:
		basename = os.path.basename(filename)
		id_action = basename[1:3]
		id_actor = basename[5:7]
		id_video = basename[9:11]

		skeleton_annotation_file = os.path.join(path_to_videos, f"a{id_action}_s{id_actor}_e{id_video}_realworld.txt")
		id_action = int(id_action)
		id_actor = int(id_actor)
		id_video = int(id_video)
		content = open(skeleton_annotation_file, "r+").readlines()
		current_sequence_id += 1
		for i in range(len(content) // 20): # We have 20 keypoints in one frame
			keypoints = []
			for i_keypoint in range(20):
				splitted_line = content[i*20 + i_keypoint].strip().split(" ")
				keypoints.append([float(splitted_line[0]),
				                  float(splitted_line[1]),
				                  float(splitted_line[2])])

			if len(X) == current_sequence_id: # First frame of an interesting sequence
				X.append([calculate_and_normalize_joints_distances(keypoints)]) # Append the keypoints of this frame
				Y.append(id_action) # And add the action label to the ground truth
			else: # Or add to the list already existing
				X[current_sequence_id].append(calculate_and_normalize_joints_distances(keypoints))

	return X, Y

def find_best_nb_poses(X, Y):
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)
	best_accuracy = 0
	best_number_of_poses = 0
	for nb_poses in range(1, 31):
		process_classification(X_train, Y_train, number_of_poses = nb_poses)
		acc = process_classification(X_test, Y_test, number_of_poses = nb_poses, make_prediction = True)['accuracy']
		print(f"With {nb_poses} : {round(acc * 100, 2)}%")
		if acc > best_accuracy:
			best_accuracy = acc
			best_number_of_poses = nb_poses

	print(f"Best -> With {best_number_of_poses} : {round(best_accuracy * 100, 2)}%")
	process_classification(X_train, Y_train, number_of_poses = best_number_of_poses)
	return process_classification(X_test, Y_test, number_of_poses = best_number_of_poses, make_prediction = True)

if __name__ == "__main__":
	# client = InfluxDB_connection.ListicClientDB(server_side = False)
	# data = from_video_annotation_to_action_list("video_annotations.json", client)
	# json.dump(data, open("action_sequences.json", 'w+'), indent = 4)

	# X, Y = process_UTKinect_datset("annotations/UTKinect-dataset", "annotations/UTKinect-dataset/UTKinect-dataset_train.txt")
	X, Y = process_Florence3D_datset("annotations/florence3D-dataset/Florence_3d_actions/Florence_dataset_WorldCoordinates.txt")
	# X, Y = process_KARD_datset("annotations/KARD-dataset/")

	find_best_nb_poses(X, Y)
