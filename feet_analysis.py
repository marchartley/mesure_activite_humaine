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


def save_data(bag_filename, saving_filename, visual = False, needed_keypoints = []):
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

			if skeletrack.check_user_presence(color_image = color_frame):
				human = human_skeleton.HumanSkeleton()
				human.from_tracker(skeletrack, 0, depth_frame)

				coord2D, coord3D = human.to_json()

				all_keypoints_valid = True
				# for keypoint in needed_keypoints:
				# 	if keypoint not in coord3D:
				# 		all_keypoints_valid = False
				if all_keypoints_valid:

					for key in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
						if key not in coordinates:
							coordinates[key] = []
							velocities[key] = []
							for key2 in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
								distances[key + "-" + key2] = []

						if key not in coord2D or np.any(coord2D[key] == -1) or np.any(coord2D[key] is None): # Coord not found
							coordinates[key].append(np.array([None, None, None]))
							velocities[key].append(None)
							for key2 in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
								distances[key + "-" + key2].append(None)
							continue

						# If coord is found
						coordinates[key].append(coord3D[key])
						if len(coordinates[key]) == 1: # Not sure this can happen
							velocities[key].append(0)
						else:
							if np.any(coordinates[key][-2] == None) or np.any(coordinates[key][-1] == None):
								velocities[key].append(None)
							else:
								velocities[key].append((np.linalg.norm((coordinates[key][-2] - coordinates[key][-1]) / (
											current_timestamp - previous_timestamp))) * 1000)

						for key2 in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
							distances[key + "-" + key2].append(None) # By default, set all to None
						for key2 in coord3D: # Change the values we can
							distances[key + "-" + key2][-1] = np.linalg.norm(coord3D[key2] - coord3D[key])

					"""
					for key in coord3D:
						if key not in coordinates:
							coordinates[key] = []
							velocities[key] = []

							for key2 in human_skeleton.BodyParts:
								distances[key + "-" + key2.name] = []

						if np.any(coord2D[key] == -1):
							coordinates[key].append(np.array([None, None, None]))
							velocities[key].append(None)
							for key2 in coord3D:
								distances[key + "-" + key2].append(None)

						# continue #coord3D[key] = np.array([None, None, None])

						coordinates[key].append(coord3D[key])
						if len(coordinates[key]) == 1:
							velocities[key].append(0)
						else:
							if np.any(coordinates[key][-2] == None) or np.any(coordinates[key][-1] == None):
								velocities[key].append(None)
							else:
								velocities[key].append(np.linalg.norm((coordinates[key][-2] - coordinates[key][-1]) / (
											current_timestamp - previous_timestamp)))

						for key2 in coord3D:
							distances[key + "-" + key2].append(np.linalg.norm(coord3D[key2] - coord3D[key]))
					"""

					timestamps.append(current_timestamp)
					frame_indexes.append(num_frame)

			if visual:
				# cv2.imshow("Camera", cv2.cvtColor(np.array(rs.colorizer().colorize(filters.l515_filter(depth_frame)).get_data()), cv2.COLOR_BGR2RGB))
				cv2.imshow("Camera", cv2.cvtColor(skeletrack.draw_skeleton(color_frame), cv2.COLOR_BGR2RGB))
				key = cv2.waitKey(1)

		except Exception as e:
			# raise(e)
			break

	# velocities = {}
	# for key in coordinates:
	# 	for i in range(len(coordinates[key])):
	# 		if coordinates[key][i] !=

	non_ndarray_coord = {}
	for key in coordinates:
		non_ndarray_coord[key] = []
		for coord in coordinates[key]:
			non_ndarray_coord[key].append(coord.tolist())

	data = {
		# 'right' : right_foot_velocity_history,
		# 'left' : left_foot_velocity_history,
		# 'distance' : feet_distance_history,
		'coordinates': non_ndarray_coord,
		'velocities': velocities,
		'distances': distances,
		'timestamps': timestamps,
		'frames': frame_indexes
	}
	# print(data)
	with open(saving_filename, 'w+') as file:
		json.dump(data, file)

	print("Done.")

	pipeline = stop_playback(pipeline)


# pipeline.stop()

def estimate_footsteps(data, motion_threshold = 10.0):
	left_data = data['left']
	right_data = data['right']
	timestamps = data['timestamps']
	frames = data['frames']
	estimated_timestamps = []
	estimated_frames = []

	number_of_steps = 0
	current_foot_walking = None
	for (left, right, timestamp, frame) in zip(left_data, right_data, timestamps, frames):
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
	return number_of_steps, {'timestamps': np.array(estimated_timestamps), 'frames': np.array(estimated_frames)}


def estimate_footsteps_by_distance(data, distance_range = 2):
	distance = data['distance']
	timestamps = data['timestamps']
	frames = data['frames']
	estimated_timestamps = []
	estimated_frames = []
	number_of_steps = 0

	for i in range(1, len(distance) - 1):
		# If dist[i] is a peak, count as a step
		# print(i, ':', np.all(distance[i - distance_range: i] < distance[i]), ' ', np.all(distance[i +1 : i + distance_range + 1] < distance[i]))
		if distance[i] > 0 and not np.any(
				distance[max(0, i - distance_range): min(len(distance), i + distance_range + 1)] > distance[
					i]):  # distance[i - 1] < distance[i] and distance[i + 1] < distance[i]:
			number_of_steps += 1
			estimated_timestamps.append(timestamps[i])
			estimated_frames.append(frames[i])

	return number_of_steps, {'timestamps': np.array(estimated_timestamps), 'frames': np.array(estimated_frames)}

def estimate_footsteps_width(data, smoothed_data):
	# nb_steps, steps_instant = estimate_footsteps(smoothed_data, 0)
	frames = []
	for i, (left_vel, right_vel) in enumerate(zip(data['velocities']['LEFT_ANKLE'], data['velocities']['RIGHT_ANKLE'])):
		if left_vel is not None and right_vel is not None:
			if left_vel < 0.01 and right_vel < 0.01:
				frames.append(data['frames'][i])
	steps_instant = {'frames': frames}
	print(frames)
	previous_positions = None
	widths = []
	number_of_measurments = 0
	for i, frame in enumerate(steps_instant['frames']):
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

	print(widths)
	return np.median(widths) # Ignore the first frame

def get_data(json_filename):
	with open(json_filename, 'r') as file:
		return json.load(file)


def truncate_values(values, limit, convert_None_to = None):
	return np.array([convert_None_to if val is None else val if val < limit else limit for val in values])


def replace_limit_values_with_previous(values, limit, convert_None_to = None):
	new_vals = []
	values = truncate_values(values, limit)
	for i in range(len(values)):
		if i == 0:
			new_vals.append(values[i])
		else:
			new_vals.append(convert_None_to if values[i] is None else values[i] if values[i] < limit else new_vals[-1])
	return np.array(new_vals)


def mean_smooth_data(values, delta):
	smoothed = []
	for i in range(len(values)):
		low_bound = max(0, i - delta)
		high_bound = min(len(values) - 1, i + delta + 1)
		try:
			bounded_values = values[low_bound:high_bound].astype(np.float)
			bounded_values = bounded_values[np.logical_and(~np.isnan(bounded_values), np.isfinite(bounded_values))]
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
	footsteps_motion = estimate_footsteps(data, motion_threshold = 0.2)[1]
	footsteps_distance = estimate_footsteps_by_distance(data, 1)[1]

	timestamps = data['timestamps']
	frames = data['frames']
	timestamps = ((timestamps - timestamps[0]))# / 1000)

	if time_in_seconds:
		x_axis = timestamps
		x_label = "Time (s)"
		motion_detection_motion = footsteps_motion['timestamps']
		motion_detection_distance = footsteps_distance['timestamps']
	else:
		x_axis = frames
		x_label = "Time (frames)"
		motion_detection_motion = footsteps_motion['frames']
		motion_detection_distance = footsteps_distance['frames']

	plt.subplot(211)
	plt.plot(x_axis, data['left'], 'r-', label = 'Left foot')
	plt.plot(x_axis, data['right'], 'b-', label = 'Right foot')
	for foot_estimation in motion_detection_motion:
		plt.axvline(x = foot_estimation, color = (.0, .0, 1.0, .4))
	plt.title("Difference of velocity per foot in time")
	plt.ylabel("Speed (m/s)")
	plt.xlabel(x_label)
	plt.legend()

	plt.subplot(212)
	plt.plot(x_axis, data['distance'] / 10, 'g-', label = 'Distance between feet')
	for foot_estimation in motion_detection_distance:
		plt.axvline(x = foot_estimation, color = (.0, .0, 1.0, .4))
	plt.ylabel('Distance (cm)')
	plt.title("Distance between feet in time")
	plt.xlabel(x_label)
	plt.legend()

	plt.show()


if __name__ == "__main__":
	# save_data("recordings/walking_slow.bag", "data_2021_03_03_slow.json", visual=True)
	# data = get_data("data_2021_03_03_normal.json")

	motion_threshold = 0.01
	# save_data("recordings_archive/20210415_150731.bag", "data_big_steps.json",
	# save_data("recordings_archive/20210415_150752.bag", "data_small_steps.json",
	#           visual = True,
	#           needed_keypoints = ['RIGHT_ANKLE', 'LEFT_ANKLE'])
	data = get_data("data_big_steps.json")
	# data = get_data("data_small_steps.json")

	left = np.array(data['velocities']['LEFT_ANKLE'])
	right = np.array(data['velocities']['RIGHT_ANKLE'])
	distance = np.array(data['distances']['RIGHT_ANKLE-LEFT_ANKLE'])


	left = remove_None_by_interpolation(left)
	right = remove_None_by_interpolation(right)
	distance = remove_None_by_interpolation(distance)

	# left = truncate_values(left, 20)
	# right = truncate_values(right, 20)
	# distance = truncate_values(distance, 500)
	# left = replace_limit_values_with_previous(left, 20)
	# right = replace_limit_values_with_previous(right, 20)
	# distance = replace_limit_values_with_previous(distance, 2000)
	left = mean_smooth_data(left, 2)
	right = mean_smooth_data(right, 2)
	distance = mean_smooth_data(distance, 2)
	left = mean_smooth_data(left, 2)
	right = mean_smooth_data(right, 2)
	distance = mean_smooth_data(distance, 2)

	timestamps = np.array(data['timestamps']) - data['timestamps'][0]  # Begin timestamps at 0
	frames = np.array(data['frames'])

	diff_right = right - left
	diff_left = left - right

	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0

	smoothed_data = {
		'right': diff_right,
		'left': diff_left,
		'distance': distance,
		'timestamps': timestamps,
		'frames': frames
	}

	# print(smoothed_data['right'])

	footsteps_motion = estimate_footsteps(smoothed_data, motion_threshold = motion_threshold)[0]
	footsteps_distance = estimate_footsteps_by_distance(smoothed_data, 1)[0]
	print(footsteps_motion, "footsteps estimated with motion")
	print(footsteps_distance, "footsteps estimated with distance")
	# for i in range(20):
	# 	print(estimate_footsteps_by_distance(smoothed_data, i)[0], "footsteps estimated with distance [i = ", i, "]")

	print("Cycle Time                 = ", footsteps_motion / ((timestamps[-1] - timestamps[0]) / 1000))
	print("Stance/Swing ratio (Right) = ", (len(diff_right) / len(diff_right[diff_right <= motion_threshold])))
	print("Stance/Swing ratio (Left)  = ", (len(diff_left) / len(diff_left[diff_left <= motion_threshold])))
	print("Double support time        = ",
	      len(diff_right[abs(diff_right - diff_left) < motion_threshold] < motion_threshold) / len(diff_right))
	print("Width of footsteps         = ", estimate_footsteps_width(data, smoothed_data))
	# dist_hip_right = (right_hip.mean() * (timestamps[-1] - timestamps[0]))/1000
	# dist_hip_left = (left_hip.mean() * (timestamps[-1] - timestamps[0]))/1000
	# print("Distance walked            = ", (dist_hip_left + dist_hip_right)/2, "m")
	# print("Distance by footstep       = ", ((dist_hip_left + dist_hip_right)/2) / footsteps_motion, "m/footstep")


	visualize_data(smoothed_data, time_in_seconds = True)

