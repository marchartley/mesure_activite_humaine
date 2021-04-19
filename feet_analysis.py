import copy

import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time
import json

import matplotlib.pyplot as plt

from convert_realsense_to_real_world import *

import human_skeleton


def get_coordinates_and_distances_from_rosbag(bag_filename, saving_filename, visual = False):
	pipeline = rs.pipeline()

	skeletrack = st.SkeletonTracker(choose_best_module = True, pipeline = pipeline)
	pipeline = start_playback(bag_filename, pipeline = pipeline, realtime = False)
	joint_confidence = 0.4

	coordinates = {}
	distances = {}
	velocities = {}
	timestamps = []
	frame_indexes = []

	num_frame = 0

	start_time = time.time()
	human = None

	current_timestamp = 0
	previous_timestamp = 0

	print("Starting...")

	while True:
		try:
			previous_timestamp = current_timestamp
			frames = pipeline.wait_for_frames(timeout_ms = 100)
			num_frame += 1
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()
			depth_frame = filters.l515_filter(depth_frame).as_depth_frame()

			current_timestamp = recording.get_frame_timestamp(frames)
			if previous_timestamp == current_timestamp:
				continue

			# if not color_frame or not depth_frame: continue

			color_frame = np.asanyarray(color_frame.get_data()).copy()

			skeletons = skeletrack.track_skeletons(color_frame, debug_time = False)

			has_human_presence = False
			if skeletrack.check_user_presence(color_image = color_frame):
				human = human_skeleton.HumanSkeleton()
				human.from_tracker(skeletrack, 0, depth_frame)

				coord2D, coord3D = human.to_json()
				has_human_presence = True

			for keypoint in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
				if keypoint not in coordinates:
					coordinates[keypoint] = []

				# Coord not found
				if not has_human_presence or keypoint not in coord2D or np.any(coord2D[keypoint] == -1) or np.any(coord2D[keypoint] is None):
					coordinates[keypoint].append(np.array([None, None, None]))
					continue
				else: # If coord is found
					coordinates[keypoint].append(coord3D[keypoint])

			timestamps.append(current_timestamp)
			frame_indexes.append(num_frame)

			if abs(current_timestamp - 1618492055246) < 100:
				print(np.linalg.norm(coordinates['LEFT_ANKLE'][-1] - coordinates['RIGHT_ANKLE'][-1]))

			if visual:
				# cv2.imshow("Camera", cv2.cvtColor(np.array(rs.colorizer().colorize(filters.l515_filter(depth_frame)).get_data()), cv2.COLOR_BGR2RGB))
				cv2.imshow("Camera", cv2.cvtColor(skeletrack.draw_skeleton(color_frame), cv2.COLOR_BGR2RGB))
				key = cv2.waitKey(1)

		except Exception as e:
			# raise(e)
			break

	# Compute velocities
	for keypoint in coordinates:
		previous_pos = np.array(coordinates[keypoint][:-1])
		next_pos = np.array(coordinates[keypoint][1:]) # Shift the array by one
		number_of_velocities = len(coordinates[keypoint]) - 1
		movement = np.array([0] * number_of_velocities)
		for i in range(number_of_velocities):
			# if keypoint == "RIGHT_ANKLE":
			# 	print(previous_pos[i], next_pos[i])
			if None in previous_pos[i] or None in next_pos[i]:
				continue
			movement[i] = (np.linalg.norm(next_pos[i] - previous_pos[i]) / (timestamps[i] - timestamps[i-1])) * 1000
		velocities[keypoint] = np.append([0], movement).tolist()
		# print(velocities[keypoint])

	# Compute distances
	for keypoint1 in human_skeleton.BodyParts:
		key1 = keypoint1.name
		for keypoint2 in human_skeleton.BodyParts:
			key2 = keypoint2.name

			for i in range(len(coordinates[key1])):
				distance = None
				if None not in coordinates[key1][i] and None not in coordinates[key2][i]:
					distance = np.linalg.norm(coordinates[key1][i] - coordinates[key2][i])
				if f"{key1}-{key2}" not in distances:
					distances[f"{key1}-{key2}"] = []
				distances[f"{key1}-{key2}"].append(distance)


	for keypoint in coordinates:
		coordinates[keypoint] = np.array(coordinates[keypoint]).tolist()

	data = {
		# 'right' : right_foot_velocity_history,
		# 'left' : left_foot_velocity_history,
		# 'distance' : feet_distance_history,
		'coordinates': coordinates,
		'velocities': velocities,
		'distances': distances,
		'timestamps': timestamps,
		'frames': frame_indexes
	}

	with open(saving_filename, 'w+') as file:
		json.dump(data, file)

	print("Done.")

	pipeline = stop_playback(pipeline)
	return data


# pipeline.stop()

def estimate_footsteps(data, motion_threshold = 0.01):

	diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
	diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0

	timestamps = data['timestamps']
	frames = data['frames']

	estimated_timestamps = []
	estimated_frames = []

	number_of_steps = 0
	current_foot_walking = None
	for (left, right, timestamp, frame) in zip(diff_left, diff_right, timestamps, frames):
		if left > motion_threshold and current_foot_walking != "left":
			current_foot_walking = "left"
			number_of_steps += 1
			estimated_timestamps.append(timestamp)
			estimated_frames.append(frame)
		elif right > motion_threshold and current_foot_walking != "right":
			current_foot_walking = "right"
			number_of_steps += 1
			estimated_timestamps.append(timestamp)
			estimated_frames.append(frame)
	return number_of_steps, {'timestamps': np.array(estimated_timestamps) - timestamps[0], 'frames': np.array(estimated_frames)}


def estimate_footsteps_by_distance(data, distance_range = 2):
	distance = data['distances']['RIGHT_ANKLE-LEFT_ANKLE']
	timestamps = data['timestamps']
	frames = data['frames']

	estimated_timestamps = []
	estimated_frames = []
	number_of_steps = 0

	for i in range(1, len(distance) - 1):
		# If dist[i] is a peak, count as a step
		if 0 < distance[i] != distance[i - 1] and not np.any(
				distance[max(0, i - distance_range): min(len(distance), i + distance_range + 1)] > distance[
					i]):
			number_of_steps += 1
			estimated_timestamps.append(timestamps[i])
			estimated_frames.append(frames[i])

	return number_of_steps, {'timestamps': np.array(estimated_timestamps) - timestamps[0], 'frames': np.array(estimated_frames)}

def estimate_footsteps_width(data):
	frames = []
	_, steps_instant = estimate_footsteps_by_distance(data, 3)
	frames += steps_instant['frames'].tolist()
	_, steps_instant = estimate_footsteps(data, motion_threshold = 0.01)
	frames += steps_instant['frames'].tolist()
	# frames = []
	# for i, (left_vel, right_vel) in enumerate(zip(data['velocities']['LEFT_ANKLE'], data['velocities']['RIGHT_ANKLE'])):
	# 	if left_vel is not None and right_vel is not None:
	# 		if left_vel < 0.01 and right_vel < 0.01:
	# 			frames.append(data['frames'][i])
	# steps_instant = {'frames': frames}
	previous_positions = None
	widths = []
	number_of_measurments = 0
	for i, frame in enumerate(frames):
		# if i < 4 or i > len(steps_instant['frames']) - 3:
		# 	continue
		number_of_measurments += 1
		index_frame = data['frames'].index(frame)
		current_left_pos = np.array(data['coordinates']['LEFT_ANKLE'][index_frame])
		current_right_pos = np.array(data['coordinates']['RIGHT_ANKLE'][index_frame])
		if previous_positions is None:
			previous_positions = {'right': current_right_pos,
			                      'left': current_left_pos}
		else:
			try:
				left_move = current_left_pos - previous_positions['left']
				right_move = current_right_pos - previous_positions['right']

				if np.linalg.norm(left_move) > np.linalg.norm(right_move): # Left foot is the one at the front
					right_pos = np.mean([previous_positions['right'], current_right_pos], axis = 0) # Do the mean because the foot might have moved a little bit
					current_width = np.linalg.norm(np.cross(left_move, current_left_pos - right_pos)) / np.linalg.norm(left_move)
				else: # Right foot is the one at the front
					left_pos = np.mean([previous_positions['left'], current_left_pos], axis = 0) # Do the mean because the foot might have moved a little bit
					current_width = np.linalg.norm(np.cross(right_move, current_right_pos - left_pos)) / np.linalg.norm(right_move)

				if current_width > 0.8 or np.isnan(current_width):
					number_of_measurments -= 1
				else:
					widths.append(current_width)
			except TypeError as e:
				pass

			if current_right_pos is not None:
				previous_positions['right'] = current_right_pos
			if current_left_pos is not None:
				previous_positions['left'] = current_left_pos

	return np.median(widths)


def get_data(json_filename):
	with open(json_filename, 'r') as file:
		return json.load(file)


def truncate_values(values, limit, convert_None_to = None):
	return np.array([convert_None_to if np.isnan(val) else val if val < limit else limit for val in values])


def replace_limit_values_with_previous(values, limit, convert_None_to = None):
	new_vals = []
	values = truncate_values(values, limit)
	for i in range(len(values)):
		if i == 0:
			new_vals.append(values[i])
		else:
			new_vals.append(convert_None_to if values[i] is None else values[i] if values[i] < limit else new_vals[-1])
	return np.array(new_vals)


def mean_smooth_data(values, delta, v=False):
	smoothed = []
	if v:
		print(len(values))
	for i in range(len(values)):
		low_bound = max(0, i - delta)
		high_bound = min(len(values) - 1, i + delta + 1)
		try:
			bounded_values = values[low_bound:high_bound].astype(np.float)
			bounded_values = bounded_values[np.logical_and(~np.isnan(bounded_values), np.isfinite(bounded_values))]
			if v:
				print(bounded_values, np.mean(bounded_values))
			smoothed.append(np.mean(bounded_values))
		except Exception as e:
			print(e)
			smoothed.append(0)
	return np.array(smoothed)


def remove_None_by_interpolation(values, extrem_values = 0):
	for i in range(len(values)):
		if values[i] is None:
			distance_before = 0
			distance_after = 0
			value_before = extrem_values
			value_after = extrem_values
			for j in range(i, 0, -1):
				if values[j] is not None:
					value_before = values[j]
					break
				else:
					distance_before += 1
			for j in range(i, len(values)):
				if values[j] is not None:
					value_after = values[j]
					break
				else:
					distance_after += 1

			values[i] = value_before * (distance_before / (distance_before + distance_after)) + value_after * (
						distance_after / (distance_before + distance_after))
	return values


def visualize_data(data, time_in_seconds = True):
	footsteps_motion = estimate_footsteps(data, motion_threshold = 0.01)[1]
	footsteps_distance = estimate_footsteps_by_distance(data, 3)[1]

	timestamps = np.array(data['timestamps'])
	frames = data['frames']
	timestamps = ((timestamps - timestamps[0]) / 1000)

	diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
	diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0

	if time_in_seconds:
		x_axis = timestamps
		x_label = "Time (s)"
		motion_detection_motion = footsteps_motion['timestamps'] / 1000
		motion_detection_distance = footsteps_distance['timestamps'] / 1000
	else:
		x_axis = frames
		x_label = "Time (frames)"
		motion_detection_motion = footsteps_motion['frames']
		motion_detection_distance = footsteps_distance['frames']

	plt.subplot(211)
	plt.plot(x_axis, diff_left, 'r-', label = 'Left foot')
	plt.plot(x_axis, diff_right, 'b-', label = 'Right foot')
	for foot_estimation in motion_detection_motion:
		plt.axvline(x = foot_estimation, color = (.0, .0, 1.0, .4))
	plt.title("Difference of velocity per foot in time")
	plt.ylabel("Speed (m/s)")
	plt.xlabel(x_label)
	plt.legend()

	plt.subplot(212)
	plt.plot(x_axis, data['distances']['RIGHT_ANKLE-LEFT_ANKLE'], 'g-', label = 'Distance between feet')
	for foot_estimation in motion_detection_distance:
		plt.axvline(x = foot_estimation, color = (.0, .0, 1.0, .4))
	plt.ylabel('Distance (m)')
	plt.title("Distance between feet in time")
	plt.xlabel(x_label)
	plt.legend()

	plt.show()

def preprocess_data_for_walk_analysis(original_data):
	data = copy.deepcopy(original_data)
	for keypoint in data['velocities']:
		vel = np.array(data['velocities'][keypoint])
		vel = remove_None_by_interpolation(vel)
		vel = mean_smooth_data(vel, 2)
		vel = mean_smooth_data(vel, 2)
		data['velocities'][keypoint] = vel

	for key_couple in data['distances']:
		distance = np.array(data['distances'][key_couple])
		distance = mean_smooth_data(distance, 2)
		distance = mean_smooth_data(distance, 2)
		distance = truncate_values(distance, 3.00, convert_None_to = -0.00001) # Consider distances >3m invalid
		data['distances'][key_couple] = distance

	return data

def reduce_walking_data_size(original_data, new_size:int):
	data = copy.deepcopy(original_data)
	for keypoint in data['velocities']:
		data['velocities'][keypoint] = data['velocities'][keypoint][:new_size]
		data['coordinates'][keypoint] = data['coordinates'][keypoint][:new_size]
		for keypoint2 in data['velocities']:
			data['distances'][f'{keypoint}-{keypoint2}'] = data['distances'][f'{keypoint}-{keypoint2}'][:new_size]
	data['frames'] = data['frames'][:new_size]
	data['timestamps'] = data['timestamps'][:new_size]

	return data

def compute_cycle_time(data, motion_threshold = 0.01):
	number_of_steps = estimate_footsteps(data, motion_threshold = motion_threshold)[0]
	sequence_duration = (data['timestamps'][-1] - data['timestamps'][0]) / 1000
	return number_of_steps / sequence_duration

def compute_stance_swing_ratio(data, motion_threshold = 0.01, concerned_foot = None):
	diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
	diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0
	if concerned_foot is None:
		return ((len(diff_right) / len(diff_right[diff_right <= motion_threshold])) +
		        (len(diff_left) / len(diff_left[diff_right <= motion_threshold]))) / 2
	elif "left" in concerned_foot.lower():
		return len(diff_left) / len(diff_left[diff_left <= motion_threshold])
	elif "right" in concerned_foot.lower():
		return len(diff_right) / len(diff_right[diff_right <= motion_threshold])
	else:
		raise Exception("Invalid foot selected to compute swing/stance ratio, use 'LEFT_ANKLE', 'RIGHT_ANKLE' or None")

def compute_double_support_time(data, motion_threshold = 0.01):
	diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
	diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0
	# Number of frames with each foot velocity < motion_threshold, divided by the length of the sequence
	return len(diff_right[abs(diff_right - diff_left) < motion_threshold] < motion_threshold) / len(diff_right)

def compute_walking_speed(data):
	return np.mean([np.mean(data['velocities']['RIGHT_SHOULDER']), np.mean(data['velocities']['LEFT_SHOULDER']),
					np.mean(data['velocities']['RIGHT_HIP']), np.mean(data['velocities']['LEFT_HIP'])])


if __name__ == "__main__":
	motion_threshold = 0.01 # 0.01 m/s ~= 0.5m/min
	# save_data("recordings/20210415_150731.bag", "data_big_steps.json",
	# save_data("recordings/20210415_150752.bag", "data_small_steps.json",
	#           visual = True,
	#           needed_keypoints = ['RIGHT_ANKLE', 'LEFT_ANKLE'])
	# data = get_data("data_big_steps.json")
	data = get_data("data_small_steps.json")

	original_data = preprocess_data_for_walk_analysis(data)

	footsteps_per_velo = []
	footsteps_per_dist = []
	cycle_times = []
	stance_swings = {'left': [], 'right': [], 'mean': []}
	double_supports = []
	widths = []
	x_axis = []
	for i in range(2, len(data['timestamps'])): # From all data, to only 2 frames
		data = reduce_walking_data_size(original_data, i)
		footsteps_per_velo.append(estimate_footsteps(data, motion_threshold = motion_threshold)[0])
		footsteps_per_dist.append(estimate_footsteps_by_distance(data)[0])
		cycle_times.append(compute_cycle_time(data, motion_threshold = motion_threshold))
		stance_swings['left'].append(compute_stance_swing_ratio(data, concerned_foot = 'left', motion_threshold = motion_threshold))
		stance_swings['right'].append(compute_stance_swing_ratio(data, concerned_foot = 'right', motion_threshold = motion_threshold))
		stance_swings['mean'].append(compute_stance_swing_ratio(data, motion_threshold = motion_threshold))
		double_supports.append(compute_double_support_time(data, motion_threshold = motion_threshold))
		widths.append(estimate_footsteps_width(data))
		x_axis.append(i)

	plt.plot(x_axis, footsteps_per_velo)
	plt.show()



	# footsteps_motion = estimate_footsteps(original_data, motion_threshold = motion_threshold)[0]
	# print(footsteps_motion, "footsteps estimated with motion")
	#
	# footsteps_distance = estimate_footsteps_by_distance(original_data, 3)[0]
	# print(footsteps_distance, "footsteps estimated with distance")
	#
	# print("Cycle Time                 = ", compute_cycle_time(original_data))
	# print("Stance/Swing ratio (Right) = ", compute_stance_swing_ratio(original_data, motion_threshold = motion_threshold,
	#                                                                   concerned_foot = "RIGHT_ANKLE"))
	# print("Stance/Swing ratio (Left)  = ", compute_stance_swing_ratio(original_data, motion_threshold = motion_threshold,
	#                                                                   concerned_foot = "LEFT_ANKLE"))
	# print("Double support time        = ", compute_double_support_time(original_data, motion_threshold = motion_threshold))
	# print("Width of footsteps         = ", estimate_footsteps_width(original_data))
	#
	#
	#
	#
	# visualize_data(original_data, time_in_seconds = True)
