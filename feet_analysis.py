import copy

import cv2
import pandas as pd
import InfluxDB_connection
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time
import json

import matplotlib.pyplot as plt

from convert_realsense_to_real_world import *

import human_skeleton


# From the skeleton, compute walking gait parameters

# Estimate the number of footsteps on the sequence by computing the velocity of each foot
def estimate_footsteps(data, motion_threshold=0.01):
    # Get the difference of velocity of each foot
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    # Remove negative values
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0

    timestamps = data['timestamps']
    frames = data['frames']

    estimated_timestamps = []
    estimated_frames = []

    number_of_steps = 0
    current_foot_walking = None
    for (left, right, timestamp, frame) in zip(diff_left, diff_right, timestamps, frames):
        # If it's the left foot that is moving, and that the movement is fast enough
        if left > motion_threshold and current_foot_walking != "left":
            current_foot_walking = "left"
            number_of_steps += 1
            # Store this frame
            estimated_timestamps.append(timestamp)
            estimated_frames.append(frame)
        # Same for the right foot
        elif right > motion_threshold and current_foot_walking != "right":
            current_foot_walking = "right"
            number_of_steps += 1
            estimated_timestamps.append(timestamp)
            estimated_frames.append(frame)
    # Return the number of footsteps found and the times where it was detected
    return number_of_steps, {'timestamps': np.array(estimated_timestamps) - timestamps[0],
                             'frames': np.array(estimated_frames)}


# Compute the number of footsteps of a sequence based on the distance between the two feet
def estimate_footsteps_by_distance(data, distance_range=2):
    # Get the distance between the two feet for each frame
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

    return number_of_steps, {'timestamps': np.array(estimated_timestamps) - timestamps[0],
                             'frames': np.array(estimated_frames)}


# Estimate the width of the gait
def estimate_footsteps_width(data):
    # It is hard to decide if we want to use the footstep estimation by velocity or by distance,
    #   so we are using the two of them here
    frames = []
    _, steps_instant = estimate_footsteps_by_distance(data, 3)
    frames += steps_instant['frames'].tolist()
    _, steps_instant = estimate_footsteps(data, motion_threshold=0.01)
    frames += steps_instant['frames'].tolist()
    previous_positions = None

    # Store all the widths founds so we can average them at the end
    widths = []
    number_of_measurments = 0  # No use for this variable
    for i, frame in enumerate(frames):
        number_of_measurments += 1
        index_frame = data['frames'].index(frame)
        # Get the position of the two feet
        current_left_pos = np.array(data['coordinates']['LEFT_ANKLE'][index_frame])
        current_right_pos = np.array(data['coordinates']['RIGHT_ANKLE'][index_frame])

        # If it's the first frame, we just store these positions
        if previous_positions is None:
            previous_positions = {'right': current_right_pos,
                                  'left': current_left_pos}
        else:
            try:
                # Compute the movement made by each foot
                left_move = current_left_pos - previous_positions['left']
                right_move = current_right_pos - previous_positions['right']

                # If the left foot moved more than the left one
                if np.linalg.norm(left_move) > np.linalg.norm(right_move):  # Left foot is the one at the front
                    # Get the mean position of the other foot because it might have moved a little bit
                    right_pos = np.mean([previous_positions['right'], current_right_pos],
                                        axis=0)
                    # Draw a line between the previous and the new positions of the left foot
                    # Compute the min distance of the right foot to the "left foot line"
                    current_width = np.linalg.norm(np.cross(left_move, current_left_pos - right_pos)) / np.linalg.norm(
                        left_move)
                else:  # Right foot is the one at the front
                    # Get the mean position of the other foot because it might have moved a little bit
                    left_pos = np.mean([previous_positions['left'], current_left_pos],
                                       axis=0)
                    # Draw a line between the previous and the new positions of the right foot
                    # Compute the min distance of the left foot to the "right foot line"
                    current_width = np.linalg.norm(np.cross(right_move, current_right_pos - left_pos)) / np.linalg.norm(
                        right_move)

                # If the distance is possible (less than 80cm looks like a good value), then store the distance
                if current_width < 0.8 and not np.isnan(current_width):
                    widths.append(current_width)
            except TypeError as e:
                pass

            # Save the current feet positions as the previous positions
            if current_right_pos is not None:
                previous_positions['right'] = current_right_pos
            if current_left_pos is not None:
                previous_positions['left'] = current_left_pos

    # Take the average of the widths computed
    return np.median(widths)


# Simply load the json file
def get_data(json_filename):
    with open(json_filename, 'r') as file:
        return json.load(file)


# Load the skeletons from InfluxDB
def get_data_from_db(sequence_id=None, time_between_sequences=20.0, server_side = True, start_time = 0, ignore_small_sequences = None):
    # limit = 200
    # Connect to InfluxDB and get all the skeleton's coordinates
    client = InfluxDB_connection.ListicClientDB(server_side = server_side)
    if sequence_id is None:
        _, skeletons3D = client.get_all_skeletons()
    else:
        _, skeletons3D = client.get_skeletons_in_splitted_sequences(sequence_index=sequence_id,
                                                                    time_between_sequences=time_between_sequences,
                                                                    start_time = start_time,
                                                                    ignore_small_sequences=ignore_small_sequences)
        # _, skeletons3D = sequences[sequence_id]

    # For now, we merge all the skeletons in one...
    # TODO: Take care of multiple skeletons
    skeletons3D = skeletons3D[0]

    timestamps = []
    keys = []
    for keypoint in skeletons3D:
        # skeletons3D[keypoint] = skeletons3D[keypoint][:limit]
        # Store all possible timestamps in the DB
        timestamps.extend([elem['time_ms'] for elem in skeletons3D[keypoint]])
        keys.extend([keypoint, keypoint + '_x', keypoint + '_y', keypoint + '_z'])
    timestamps = sorted(list(set(timestamps)))

    # Create the empty DataFrame with one line per possible timestamp
    full_df = pd.DataFrame(columns=keys, index=timestamps)
    for keypoint in skeletons3D:
        # Store the coordinates of this keypoint using the timestamp as index
        vals = {c['time_ms']: ([c['x'], c['y'], c['z']]) for c in skeletons3D[keypoint]}
        # Fill each cell, line-per-line, if the data is available
        for timestamp in timestamps:
            if timestamp in vals:
                # full_df[keypoint][timestamp] = vals[timestamp]
                full_df[f'{keypoint}_x'][timestamp] = vals[timestamp][0]
                full_df[f'{keypoint}_y'][timestamp] = vals[timestamp][1]
                full_df[f'{keypoint}_z'][timestamp] = vals[timestamp][2]

    # Convert the X, Y, Z's columns to numeric data
    cols = [col for col in full_df.columns if '_x' in col or '_y' in col or '_z' in col]
    for col in cols:
        full_df[col] = pd.to_numeric(full_df[col])
    # Interpolate the X, Y, Z values
    full_df[cols] = full_df[cols].interpolate(axis=0, limit_area='inside', method='linear')
    # Convert the X, Y, Z columns into vectors
    for keypoint in skeletons3D:
        full_df[keypoint] = full_df.apply(
            lambda row: np.array([row[f'{keypoint}_x'], row[f'{keypoint}_y'], row[f'{keypoint}_z']]) if not np.isnan(
                row[f'{keypoint}_x']) else np.array([0, 0, 0])
            , axis=1)

    json_data = {
        'frames': list(range(full_df.shape[0])),
        'timestamps': full_df.index.values.tolist(),
        'coordinates': {},
        'distances': {},
        'velocities': {}
    }
    for keypoint in skeletons3D:
        # The vectors have already been computed
        json_data['coordinates'][keypoint] = full_df[keypoint].values

        # Compute velocities by shifting the values and compute the difference
        previous_pos = np.array(json_data['coordinates'][keypoint][:-1])
        next_pos = np.array(json_data['coordinates'][keypoint][1:])  # Shift the array by one
        number_of_velocities = len(json_data['coordinates'][keypoint]) - 1
        movement = np.array([0] * number_of_velocities, dtype=float)
        for i in range(number_of_velocities):
            if previous_pos[i] is np.nan or next_pos[i] is np.nan or None in previous_pos[i] or None in next_pos[i]:
                continue
            try:
                movement[i] = (np.linalg.norm(next_pos[i] - previous_pos[i]) / (
                        json_data['timestamps'][i] - json_data['timestamps'][i - 1]))
            except:
                continue
        json_data['velocities'][keypoint] = np.append([0], movement)
        for keypoint2 in skeletons3D:
            json_data['distances'][f'{keypoint}-{keypoint2}'] = full_df.apply(
                lambda row: np.linalg.norm(row[keypoint] - row[keypoint2]), axis=1).values

    return json_data


# From a collection of skeletons created for example by the HumanStorage,
# Create the json data that is used in the feet analysis functions
# The "human_collection" must be in the shape:
# {
#   'timestamp1': HumanSkeleton(),
#   'timestamp2': HumanSkeleton(),
#   ...
# }
def create_feet_analysis_data_from_humans(human_collection):
    timestamps = list(human_collection.keys())
    keys = []
    full_keys = []
    for keypoint in human_skeleton.BodyParts:
        keys.append(keypoint.name)
        full_keys.extend([keypoint.name + '_' + coord for coord in ('x', 'y', 'z')])
    full_keys.extend(keys)

    # Create the empty DataFrame with one line per possible timestamp
    full_df = pd.DataFrame(columns=full_keys, index=timestamps)
    for timestamp in human_collection:
        human = human_collection[timestamp]
        for keypoint in keys:
            # Store the coordinates of this keypoint using the timestamp as index
            if human.check_keypoints3D(keypoint):
                keypoint_coord = human.get_keypoint3D(keypoint)
                # Fill each cell, line-per-line
                full_df[f'{keypoint}_x'][timestamp] = keypoint_coord[0]
                full_df[f'{keypoint}_y'][timestamp] = keypoint_coord[1]
                full_df[f'{keypoint}_z'][timestamp] = keypoint_coord[2]

    # Convert the X, Y, Z's columns to numeric data
    cols = [col for col in full_df.columns if '_x' in col or '_y' in col or '_z' in col]
    for col in cols:
        full_df[col] = pd.to_numeric(full_df[col])
    # Interpolate the X, Y, Z values
    full_df[cols] = full_df[cols].interpolate(axis=0, limit_area='inside')
    # Convert the X, Y, Z columns into vectors
    for keypoint in keys:
        full_df[keypoint] = full_df.apply(
            lambda row: np.array([row[f'{keypoint}_x'], row[f'{keypoint}_y'], row[f'{keypoint}_z']])
            if row[f'{keypoint}_x'] is not None and not np.isnan(row[f'{keypoint}_x']) else np.array([0, 0, 0])
            , axis=1)

    json_data = {
        'frames': list(range(full_df.shape[0])),
        'timestamps': full_df.index.values / 1000,
        'coordinates': {},
        'distances': {},
        'velocities': {}
    }
    for keypoint in keys:
        # The vectors have already been computed
        json_data['coordinates'][keypoint] = full_df[keypoint].values

        # Compute velocities by shifting the values and compute the difference
        previous_pos = np.array(json_data['coordinates'][keypoint][:-1])
        next_pos = np.array(json_data['coordinates'][keypoint][1:])  # Shift the array by one
        number_of_velocities = len(json_data['coordinates'][keypoint]) - 1
        movement = np.array([0] * number_of_velocities, dtype=float)
        for i in range(number_of_velocities):
            if previous_pos[i] is np.nan or next_pos[i] is np.nan or None in previous_pos[i] or None in next_pos[i]:
                continue
            try:
                movement[i] = (np.linalg.norm(next_pos[i] - previous_pos[i]) / (
                        json_data['timestamps'][i] - json_data['timestamps'][i - 1]))
            except:
                continue
        json_data['velocities'][keypoint] = np.append([0], movement)
        for keypoint2 in keys:
            json_data['distances'][f'{keypoint}-{keypoint2}'] = full_df.apply(
                lambda row: np.linalg.norm(row[keypoint] - row[keypoint2]), axis=1).values

    return json_data


# Truncate values to a certain limit. It's possible to assign a value to None values
def truncate_values(values, limit, convert_None_to=None):
    return np.array([convert_None_to if np.isnan(val) else val if val < limit else limit for val in values])


# Truncate data by replacing the elements above the limit with the previous value of the series
def replace_limit_values_with_previous(values, limit, convert_None_to=None):
    new_values = []
    values = truncate_values(values, limit)
    for i in range(len(values)):
        if i == 0:
            new_values.append(values[i])
        else:
            new_values.append(
                convert_None_to if values[i] is None else values[i] if values[i] < limit else new_values[-1])
    return np.array(new_values)


# Smooth the curve with a sliding window of size "delta"
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


# Replace None values by interpolating with surrounding values
# If there is no not-None surrounding a point, it will use the "extreme_value" for the interpolation
# This function is a very inefficient and should be replaced with a numpy/scipy function if you know a good one
def remove_None_by_interpolation(values, extreme_values=0):
    for i in range(len(values)):
        if values[i] is None:
            distance_before = 0
            distance_after = 0
            value_before = extreme_values
            value_after = extreme_values
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

def replace_trailing_and_leading_zeros_in_array(array, remove_leading = True, remove_trailing = True,
                                                replace_value = None):
    array = np.array(array)
    zeros_index = np.where(array == 0)
    if len(zeros_index[0]) == 0:
        return array  # Nothing to do
    # Group the indices that are following each other together
    # Ex: for zeros_index = [0, 1, 2, 5, 6, 10, 11]
    #   Result is [(0, 1, 2), (5, 6), (10, 11)]
    zeros_index = np.split(zeros_index[0], np.where(np.diff(zeros_index[0]) > 1)[0] + 1)
    # The first indices are the leading ones, while the last ones are the trailing 0's
    if remove_leading:
        # The first index of the first group must be 0 if it's a leading item
        if zeros_index[0][0] == 0:
            array[zeros_index[0]] = replace_value
    if remove_trailing:
        # The last index of the last group must be the last index of the original array to be "trailing"
        if zeros_index[-1][-1] == len(array) - 1:
            array[zeros_index[-1]] = replace_value
    return array

# Display a plot with the calculation of footsteps by distance and by velocity
def visualize_data(data, time_in_seconds=True, non_blocking = False):
    footsteps_motion = estimate_footsteps(data, motion_threshold=0.01)[1]
    footsteps_distance = estimate_footsteps_by_distance(data, 3)[1]

    timestamps = np.array(data['timestamps'])
    frames = data['frames']
    timestamps = timestamps - timestamps[0]

    feet_distance = data['distances']['RIGHT_ANKLE-LEFT_ANKLE']
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0

    # Replace the leading and trailing 0's to None to remove them from the plots
    diff_left = replace_trailing_and_leading_zeros_in_array(diff_left)
    diff_right = replace_trailing_and_leading_zeros_in_array(diff_right)
    feet_distance = replace_trailing_and_leading_zeros_in_array(feet_distance)

    x_start = min(np.where(diff_left != None)[0][0], np.where(diff_right != None)[0][0], np.where(feet_distance != None)[0][0])
    x_end = max(np.where(diff_left != None)[0][-1], np.where(diff_right != None)[0][-1], np.where(feet_distance != None)[0][-1])

    # Change the X-axis depending on the unit we want (frame number of time)
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

    x_axis = x_axis[x_start:x_end]
    diff_left = diff_left[x_start:x_end]
    diff_right = diff_right[x_start:x_end]
    feet_distance = feet_distance[x_start:x_end]

    # Display footsteps by velocity
    plt.subplot(211)
    plt.plot(x_axis, diff_left, 'r-', label='Left foot')
    plt.plot(x_axis, diff_right, 'b-', label='Right foot')
    for foot_estimation in motion_detection_motion:
        plt.axvline(x=foot_estimation, color=(.0, .0, 1.0, .4))
    plt.title("Difference of velocity per foot in time")
    plt.ylabel("Speed (m/s)")
    plt.xlabel(x_label)
    plt.legend()

    # Display footsteps by distance
    plt.subplot(212)
    plt.plot(x_axis, feet_distance, 'g-', label='Distance between feet')
    for foot_estimation in motion_detection_distance:
        plt.axvline(x=foot_estimation, color=(.0, .0, 1.0, .4))
    plt.ylabel('Distance (m)')
    plt.title("Distance between feet in time")
    plt.xlabel(x_label)
    plt.legend()

    plt.show(block = not non_blocking)


# Process the data to remove None values and smooth the values
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
        distance = truncate_values(distance, 3.00, convert_None_to=-0.00001)  # Consider distances >3m invalid
        data['distances'][key_couple] = distance

    return data


# Reduce the size of the keypoint coords/velocities/distances dataset
def reduce_walking_data_size(original_data, new_size: int):
    data = copy.deepcopy(original_data)
    for keypoint in data['velocities']:
        data['velocities'][keypoint] = data['velocities'][keypoint][:new_size]
        data['coordinates'][keypoint] = data['coordinates'][keypoint][:new_size]
        for keypoint2 in data['velocities']:
            data['distances'][f'{keypoint}-{keypoint2}'] = data['distances'][f'{keypoint}-{keypoint2}'][:new_size]
    data['frames'] = data['frames'][:new_size]
    data['timestamps'] = data['timestamps'][:new_size]

    return data


# Get the Cycle time in seconds = duration of the sequence / number of footsteps
def compute_cycle_time(data, motion_threshold=0.01):
    number_of_steps, times_of_steps = estimate_footsteps(preprocess_data_for_walk_analysis(data),
                                                         motion_threshold=motion_threshold)
    # Solution 1: Get the moment of the first and last step, and divide by the number of steps
    # Remove 1 step because the last timestamp is the beggining of a step, not a complete step
    sequence_duration = times_of_steps['timestamps'][-1] - times_of_steps['timestamps'][0]
    return sequence_duration / (number_of_steps - 1)
    # Solution 2: Take the first and last frame where the feet are moving
    # This doesn't take in consideration that the last frames might not be real steps
    # timestamps_used = np.array(data['timestamps'])[
    #     np.where(np.logical_and(data['velocities']['LEFT_ANKLE'] > 0, data['velocities']['RIGHT_ANKLE'] > 0))]
    # sequence_duration = timestamps_used[-1] - timestamps_used[0]
    # Solution 3: Take the beginning and ending of the full sequence.
    # sequence_duration = data['timestamps'][-1] - data['timestamps'][0]
    # return sequence_duration / number_of_steps


# Compute the Stance/Swing ratio = time with two feet down / time with one foot moving
# TODO: Improve it by trimming before the first step / after last footstep
def compute_stance_ratio(data, motion_threshold=0.01, concerned_foot=None):
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0

    static_right = np.logical_and(0 < data['velocities']['RIGHT_ANKLE'],
                                  data['velocities']['RIGHT_ANKLE'] < motion_threshold)
    available_right = data['velocities']['RIGHT_ANKLE'] > 0

    static_left = np.logical_and(0 < data['velocities']['LEFT_ANKLE'],
                                 data['velocities']['LEFT_ANKLE'] < motion_threshold)
    available_left = data['velocities']['LEFT_ANKLE'] > 0

    if concerned_foot is None:
        return np.mean([compute_stance_ratio(data=data, motion_threshold=motion_threshold, concerned_foot='right'),
                        compute_stance_ratio(data=data, motion_threshold=motion_threshold, concerned_foot='left')])
    elif "left" in concerned_foot.lower():
        return np.sum(static_left) / np.sum(available_left)
    elif "right" in concerned_foot.lower():
        return np.sum(static_right) / np.sum(available_right)
    else:
        raise Exception("Invalid foot selected to compute stance ratio, use 'LEFT_ANKLE', 'RIGHT_ANKLE' or None")


# Compute the Stance/Swing ratio = time with two feet down / time with one foot moving
# TODO: Improve it by trimming before the first step / after last footstep
def compute_stance_swing_ratio(data, motion_threshold=0.01, concerned_foot=None):
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0

    static_right = np.logical_and(0 < data['velocities']['RIGHT_ANKLE'],
                                  data['velocities']['RIGHT_ANKLE'] < motion_threshold)
    available_right = data['velocities']['RIGHT_ANKLE'] > 0

    static_left = np.logical_and(0 < data['velocities']['LEFT_ANKLE'],
                                 data['velocities']['LEFT_ANKLE'] < motion_threshold)
    available_left = data['velocities']['LEFT_ANKLE'] > 0

    static_both = np.logical_and(static_right, static_left)
    available_both = np.logical_and(available_right, available_left)

    if concerned_foot is None:
        if len(diff_right[diff_right <= motion_threshold]) == 0 or len(diff_left[diff_left <= motion_threshold]) == 0:
            return 0
        return ((len(diff_right) / len(diff_right[diff_right <= motion_threshold])) +
                (len(diff_left) / len(diff_left[diff_right <= motion_threshold]))) / 2
    elif "left" in concerned_foot.lower():
        if len(diff_left[diff_left <= motion_threshold]) == 0:
            return 0
        return np.sum(static_left) / np.sum(available_left)
    elif "right" in concerned_foot.lower():
        if len(diff_right[diff_right <= motion_threshold]) == 0:
            return 0
        return np.sum(static_right) / np.sum(available_right)
        # return len(diff_right) / len(diff_right[diff_right <= motion_threshold])
    else:
        raise Exception("Invalid foot selected to compute swing/stance ratio, use 'LEFT_ANKLE', 'RIGHT_ANKLE' or None")


# Compute Double Support Time = time with the two feet on the ground
# TODO: Improve it by trimming before the first step / after last footstep
def compute_double_support_time(data, motion_threshold=0.01):
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0
    # Number of frames with each foot velocity < motion_threshold, divided by the length of the sequence
    static_right = np.logical_and(0 < data['velocities']['RIGHT_ANKLE'],
                                  data['velocities']['RIGHT_ANKLE'] < motion_threshold)
    available_right = data['velocities']['RIGHT_ANKLE'] > 0

    static_left = np.logical_and(0 < data['velocities']['LEFT_ANKLE'],
                                 data['velocities']['LEFT_ANKLE'] < motion_threshold)
    available_left = data['velocities']['LEFT_ANKLE'] > 0

    static_both = np.logical_and(static_right, static_left)
    available_both = np.logical_and(available_right, available_left)
    return np.sum(static_both) / np.sum(available_both)
    # return len(diff_right[diff_right[abs(diff_right - diff_left) < motion_threshold] < motion_threshold]) / len(diff_right[diff_right > 0])


# Get the walking speed with the average of the movement of the shoulders/hips
# This function should maybe not be used...
def compute_walking_speed(data):
    velocities_rh = np.array(data['velocities']['RIGHT_HIP'])
    velocities_rh = velocities_rh[np.logical_and(np.isfinite(velocities_rh), velocities_rh > 0)]
    velocities_lh = np.array(data['velocities']['LEFT_HIP'])
    velocities_lh = velocities_lh[np.logical_and(np.isfinite(velocities_lh), velocities_lh > 0)]
    return np.median([np.median(velocities_lh), np.median(velocities_rh)])

    """
    velocities_rs = np.array(data['velocities']['RIGHT_SHOULDER'])
    velocities_rs = velocities_rs[np.logical_and(np.isfinite(velocities_rs), velocities_rs > 0)]
    velocities_ls = np.array(data['velocities']['LEFT_SHOULDER'])
    velocities_ls = velocities_ls[np.logical_and(np.isfinite(velocities_ls), velocities_ls > 0)]
    velocities_rh = np.array(data['velocities']['RIGHT_HIP'])
    velocities_rh = velocities_rh[np.logical_and(np.isfinite(velocities_rh), velocities_rh > 0)]
    velocities_lh = np.array(data['velocities']['LEFT_HIP'])
    velocities_lh = velocities_lh[np.logical_and(np.isfinite(velocities_lh), velocities_lh > 0)]
    return np.median([np.median(velocities_lh), np.median(velocities_rh),
                      np.median(velocities_rs), np.median(velocities_ls)])"""


# Estimate the width of the gait
def compute_stride_length(data):
    # It is hard to decide if we want to use the footstep estimation by velocity or by distance,
    #   so we are using the two of them here
    frames = []
    _, steps_instant = estimate_footsteps_by_distance(data, 3)
    frames += steps_instant['frames'].tolist()
    # _, steps_instant = estimate_footsteps(data, motion_threshold=0.01)
    # frames += steps_instant['frames'].tolist()
    previous_positions = None

    # Store all the widths founds so we can average them at the end
    stride_lengths = {'right': [], 'left': []}
    for i, frame in enumerate(frames):
        index_frame = data['frames'].index(frame)
        # Get the position of the two feet
        current_left_pos = np.array(data['coordinates']['LEFT_ANKLE'][index_frame])
        current_right_pos = np.array(data['coordinates']['RIGHT_ANKLE'][index_frame])

        # If it's the first frame, we just store these positions
        if previous_positions is None:
            previous_positions = {'right': current_right_pos,
                                  'left': current_left_pos}
        else:
            try:
                # Compute the movement made by each foot
                left_move = current_left_pos - previous_positions['left']
                right_move = current_right_pos - previous_positions['right']

                # If the left foot moved more than the left one
                if np.linalg.norm(left_move) > np.linalg.norm(right_move):  # Left foot is the one at the front
                    # Get the mean position of the other foot because it might have moved a little bit
                    right_pos = np.mean([previous_positions['right'], current_right_pos],
                                        axis=0)
                    # Draw a line between the previous and the new positions of the left foot
                    # Compute the min distance of the right foot to the "left foot line"
                    current_width = np.linalg.norm(np.cross(left_move, current_left_pos - right_pos)) / np.linalg.norm(
                        left_move)
                    stride_lengths['left'].append(np.linalg.norm(left_move))
                else:  # Right foot is the one at the front
                    # Get the mean position of the other foot because it might have moved a little bit
                    left_pos = np.mean([previous_positions['left'], current_left_pos],
                                       axis=0)
                    # Draw a line between the previous and the new positions of the right foot
                    # Compute the min distance of the left foot to the "right foot line"
                    current_width = np.linalg.norm(np.cross(right_move, current_right_pos - left_pos)) / np.linalg.norm(
                        right_move)
                    stride_lengths['right'].append(np.linalg.norm(right_move))
            except TypeError as e:
                pass

            # Save the current feet positions as the previous positions
            if current_right_pos is not None:
                previous_positions['right'] = current_right_pos
            if current_left_pos is not None:
                previous_positions['left'] = current_left_pos

    right_length = np.median(stride_lengths['right'])
    left_length = np.median(stride_lengths['left'])
    stride_length = np.mean([right_length, left_length])
    return stride_length, right_length, left_length


#
# def compute_stride_length(data):
#     walking_speed = compute_walking_speed(data)
#     number_of_steps, times_of_steps = estimate_footsteps(preprocess_data_for_walk_analysis(data),
#                                                          motion_threshold=motion_threshold)
#     # Solution 1: Get the moment of the first and last step, and divide by the number of steps
#     # Remove 1 step because the last timestamp is the beggining of a step, not a complete step
#     sequence_duration = times_of_steps['timestamps'][-1] - times_of_steps['timestamps'][0]
#     total_distance_walked = sequence_duration * walking_speed
#     return total_distance_walked / (number_of_steps - 1)

def get_walking_direction(data):
    starting_point = None
    ending_point = None
    for i in range(len(data['frames'])):
        if starting_point is None:
            starting_human = human_skeleton.HumanSkeleton()
            starting_human.from_feet_analytics(data['coordinates'], i)
            if starting_human.get_number_of_visible_keypoints() > 2:
                starting_point = starting_human.get_mean_position_of_points()
        if ending_point is None:
            ending_human = human_skeleton.HumanSkeleton()
            ending_human.from_feet_analytics(data['coordinates'], len(data['frames']) - i - 1)
            if ending_human.get_number_of_visible_keypoints() > 2:
                ending_point = ending_human.get_mean_position_of_points()
        if starting_point is not None and ending_point is not None:
            break

    # If the displacement is at least 2m away from the camera (North - South), set to 1
    if (ending_point - starting_point)[2] > 2.0:
        return 1
    # If the displacement is at least 2m towards the camera (South - North), set to -1
    elif (ending_point - starting_point)[2] < -2.0:
        return -1
    # Otherwise, the movement is quite undefined, set to 0 (this can be a way to filter after)
    else:
        return 0

def get_corridor_used(data):
    mean_pos = []
    percent_stripped = 0.2
    for i in range(len(data['frames'])):
        human = human_skeleton.HumanSkeleton().from_feet_analytics(data['coordinates'], i)
        if human.get_center_of_gravity() is not None:
            mean_pos.append(human.get_center_of_gravity())

    stripped_part = round(len(mean_pos) * percent_stripped)
    position = np.median(mean_pos[stripped_part: -stripped_part], axis = 0)

    # Estimate the corridor used by the X position
    x_pos = position[0]
    # If it's on the West side, code it as 1
    if -2.5 <= x_pos <= -1.6:
        return 1
    # If it's in the center, code it as 0
    if -1.6 <= x_pos <= -1.0:
        return 0
    # If it's on the East side, code it as -1
    if -1.6 <= x_pos <= 0.0:
        return -1


def get_walking_features(data = None, id_sequence = None, time_between_sequences = 1.0, server_side = True, motion_threshold = 0.8,
                         start_time = 0, ignore_small_sequences = None):
    if data is None:
        data = get_data_from_db(id_sequence, time_between_sequences = time_between_sequences, server_side = server_side,
                                start_time = start_time, ignore_small_sequences = ignore_small_sequences)
    original_data = preprocess_data_for_walk_analysis(data)

    footsteps_motion = estimate_footsteps(original_data, motion_threshold=motion_threshold)[0]
    footsteps_distance = estimate_footsteps_by_distance(original_data, 3)[0]
    cycle_time = compute_cycle_time(data)
    cadence = 60 / cycle_time
    stance_ratio_right = compute_stance_ratio(data, motion_threshold=motion_threshold, concerned_foot="RIGHT_ANKLE")
    stance_ratio_left = compute_stance_ratio(data, motion_threshold=motion_threshold, concerned_foot="LEFT_ANKLE")
    double_support_ratio = compute_double_support_time(data, motion_threshold=motion_threshold)
    footstep_width = estimate_footsteps_width(data)
    walking_speed = compute_walking_speed(data)
    stride_length, right_length, left_length = compute_stride_length(data)
    direction = get_walking_direction(data)
    corridor = get_corridor_used(data)

    params = {'nb_steps_motion': footsteps_motion,
              'nb_steps_distance': footsteps_distance,
              'cycle_time': cycle_time,
              'cadence': cadence,
              'stance_ratio_right': stance_ratio_right,
              'stance_ratio_left': stance_ratio_left,
              'double_support_ratio': double_support_ratio,
              'footstep_width': footstep_width,
              'walking_speed': walking_speed,
              'stride_length': stride_length,
              'stride_length_right': right_length,
              'stride_length_left': left_length,
              'direction': direction,
              'corridor': corridor,
              }
    return params

def display_walking_features(walking_features: dict):
    text_length = max([len(key) for key in walking_features])
    for key in walking_features:
        print(f"{str(key)}{' ' * (text_length - len(key))}: {walking_features[key]}")

def main():
    # motion_threshold = 0.01  # 0.01 m/s ~= 0.5m/min
    motion_threshold = 0.8  # 0.8 m/s ~= 3km/h
    # save_data("recordings/20210415_150731.bag", "data_big_steps.json",
    # save_data("recordings/20210415_150752.bag", "data_small_steps.json",
    #           visual = True,
    #           needed_keypoints = ['RIGHT_ANKLE', 'LEFT_ANKLE'])
    # data = get_data("data_big_steps.json")
    data = get_data_from_db(-1, time_between_sequences=1.0, server_side=True)
    original_data = preprocess_data_for_walk_analysis(data)

    # footsteps_per_velo = []
    # footsteps_per_dist = []
    # cycle_times = []
    # stance_swings = {'left': [], 'right': [], 'mean': []}
    # double_supports = []
    # widths = []
    # x_axis = []
    # # Compute all the parameters by changing the data size to get the evolution in time
    # for i in range(2, len(data['timestamps'])):  # From all data, to only 2 frames
    #     data = reduce_walking_data_size(original_data, i)
    #     footsteps_per_velo.append(estimate_footsteps(data, motion_threshold=motion_threshold)[0])
    #     footsteps_per_dist.append(estimate_footsteps_by_distance(data)[0])
    #     cycle_times.append(compute_cycle_time(data, motion_threshold=motion_threshold))
    #     stance_swings['left'].append(
    #         compute_stance_swing_ratio(data, concerned_foot='left', motion_threshold=motion_threshold))
    #     stance_swings['right'].append(
    #         compute_stance_swing_ratio(data, concerned_foot='right', motion_threshold=motion_threshold))
    #     stance_swings['mean'].append(compute_stance_swing_ratio(data, motion_threshold=motion_threshold))
    #     double_supports.append(compute_double_support_time(data, motion_threshold=motion_threshold))
    #     widths.append(estimate_footsteps_width(data))
    #     x_axis.append(i)
    #
    # plt.plot(x_axis, widths)
    # plt.show()

    display_walking_features(get_walking_features(data = data))
    # print(get_walking_direction(data))
    exit()

    footsteps_motion = estimate_footsteps(original_data, motion_threshold=motion_threshold)[0]
    footsteps_distance = estimate_footsteps_by_distance(original_data, 3)[0]
    cycle_time = compute_cycle_time(data)
    stance_ratio_right = compute_stance_ratio(data, motion_threshold=motion_threshold, concerned_foot="RIGHT_ANKLE")
    stance_ratio_left = compute_stance_ratio(data, motion_threshold=motion_threshold, concerned_foot="LEFT_ANKLE")
    double_support_ratio = compute_double_support_time(data, motion_threshold=motion_threshold)
    footstep_width = estimate_footsteps_width(data)
    walking_speed = compute_walking_speed(data)
    stride_length, right_length, left_length = compute_stride_length(data)

    print(footsteps_motion, "footsteps estimated with motion")
    print(footsteps_distance, "footsteps estimated with distance")

    print(f"Walking speed              = {round(walking_speed * 3.6, 2)} km/h")
    print(f"Cycle Time                 = {round(cycle_time, 3)} sec/step")
    print(f"Cadence                    = {round(60 / cycle_time, 1)} steps/min")
    print(f"Stance ratio (Right)       = {round(stance_ratio_right * 100, 2)}%")
    print(f"Stance ratio (Left)        = {round(stance_ratio_left * 100, 2)}%")
    print(f"Double support time        = {round(double_support_ratio * 100, 2)}%")
    print(f"Width of footsteps         = {round(footstep_width * 100)}cm")
    print(f"Stride length              = {round(stride_length * 100)}cm")
    print(f"Stride length (R)           = {round(right_length * 100)}cm")
    print(f"Stride length (L)          = {round(left_length * 100)}cm")

    visualize_data(original_data, time_in_seconds=True)


if __name__ == "__main__":
    main()
