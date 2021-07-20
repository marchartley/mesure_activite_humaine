"""! This script enables to estimate multiple characteristics of a person's walk through the depth camera """

import copy
from typing import Any

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


def estimate_footsteps(data: Dict, motion_threshold: float = 0.01) -> Tuple[int, Dict]:
    """! Estimate the number of footsteps on the sequence by computing the velocity of each foot
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param motion_threshold The needed motion speed of a keypoint to be considered in motion (in m/s)

    @return A tuple containing the number of steps counted and a dictionary with the list of frames index
    (and timestamps) at which the step is detected
    """
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


def estimate_footsteps_by_distance(data: Dict, window_size: int = 3) -> Tuple[int, Dict]:
    """! Compute the number of footsteps of a sequence based on the distance between the two feet
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param window_size The size of the window used to check if a frame has the maximal distance between the feet.
    With the precision of the camera we find "noise" making small peaks, so this parameter can reduce them.
    But the preprocessing of data should smooth the values and avoid using this parameter, but anyway...

    @return A tuple containing the number of steps counted and a dictionary with the list of frames index
    (and timestamps) at which the step is detected
    """
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
                distance[max(0, i - window_size): min(len(distance), i + window_size + 1)] > distance[
                    i]):
            number_of_steps += 1
            estimated_timestamps.append(timestamps[i])
            estimated_frames.append(frames[i])

    return number_of_steps, {'timestamps': np.array(estimated_timestamps) - timestamps[0],
                             'frames': np.array(estimated_frames)}


def estimate_footsteps_width(data: Dict) -> float:
    """! Estimate the stride width, which is the distance between the feet when walking
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return The mean width computed during the entire sequence.
    """
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
    return float(np.median(widths))


def get_data(json_filename: str):
    """! Simply load a json file containing the skeleton coordinates during a sequence.
    @deprecated This was used when the InfluxDB database wasn't set up yet. This is now useless I guess.
    """
    with open(json_filename, 'r') as file:
        return json.load(file)


def get_data_from_db(sequence_id: int = None, time_between_sequences: float = 20.0, server_side: bool = True,
                     start_time: int = 0, ignore_small_sequences: float = None) -> Dict:
    """! Load the skeletons from InfluxDB by providing the sequence ID and transforms the data to allow
    estimations of the gait parameters
    @param sequence_id The identifier of the sequence, use -1 to get last sequence (or -2 for previous one, etc...).
    @param time_between_sequences The number of sequence needed between two recordings to consider them as two
    distinct sequences. See the process in ListicClientDB.get_splitted_sequences_boundaries().
    @param server_side Set to True to access directly to the database in the Listic server, or False to access your
    local instance of InfluxDB.
    @param start_time The date at which you consider the start of the sequences, as timestamp in milliseconds.
    @param ignore_small_sequences If set, any sequence with a duration lower than this value (in s) will be ignored.

    @return A dictionary containing all the coordinates of a person during a sequence.
    """
    # limit = 200
    # Connect to InfluxDB and get all the skeleton's coordinates
    client = InfluxDB_connection.ListicClientDB(server_side=server_side)
    if sequence_id is None:
        _, skeletons3D = client.get_all_skeletons()
    else:
        _, skeletons3D = client.get_skeletons_in_splitted_sequences(sequence_index=sequence_id,
                                                                    time_between_sequences=time_between_sequences,
                                                                    start_time=start_time,
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


def create_feet_analysis_data_from_humans(human_collection: Dict) -> Dict:
    """! From a collection of skeletons created for example by the HumanStorage,
    Create the json data that is used in the feet analysis functions
    The "human_collection" must be in the shape:
    {
      'timestamp1': human_skeleton.HumanSkeleton(),
      'timestamp2': human_skeleton.HumanSkeleton(),
      ...
    }
    @param human_collection A dictionary with HumanSkeleton objects with the timestamp as the keys.

    @note Note
    I think this function is never used, but could be used with direct_camera_to_db.HumanStorage, I guess.
    """
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


def truncate_values(values: Union[List, np.ndarray], limit: float, convert_None_to: Any = None) -> np.ndarray:
    """! Truncate values to a certain limit. It's possible to assign a value to None values
    @param values The series to truncate
    @param limit The maximal value desired in the output series
    @param convert_None_to The value to give to "None" values. By default it doesn't modify the values.

    @return The truncated series.
    """
    return np.array([convert_None_to if np.isnan(val) else val if val < limit else limit for val in values])


def replace_limit_values_with_previous(values: List, limit: float, convert_None_to: Any = None) -> np.ndarray:
    """! Truncate data by replacing the elements above the limit with the previous value of the series
    @param values The series to truncate
    @param limit The maximal value desired in the output series
    @param convert_None_to The value to give to "None" values. By default it doesn't modify the values.

    @return The modified series.
    """
    new_values = []
    # Truncate values
    values = truncate_values(values, limit)
    for i in range(len(values)):
        # If it's the first element, we cannot give the previous element, so we just add the current one
        if i == 0:
            new_values.append(values[i])
        else:
            # If the value is the limit, that means it has been truncated, so we replace it with the previous of
            # the series
            new_values.append(
                convert_None_to if values[i] is None else values[i] if values[i] < limit else new_values[-1])
    return np.array(new_values)


def mean_smooth_data(values: Union[List, np.ndarray], windows_size: int) -> np.ndarray:
    """! Smooth the curve with a sliding window of size "windows_size"
    @param values The series to smooth
    @param windows_size The size of the window used to average the data.

    @return A smoothed version of the data using the mean average at each point.
    """
    smoothed = []
    for i in range(len(values)):
        # Compute the low and high bounds to stay in scope of the array
        low_bound = max(0, i - windows_size)
        high_bound = min(len(values) - 1, i + windows_size + 1)
        try:
            # Extract the values in the window sight
            bounded_values = values[low_bound:high_bound].astype(np.float)
            # Remove any undesired values (infinite and NaN values)
            bounded_values = bounded_values[np.logical_and(~np.isnan(bounded_values), np.isfinite(bounded_values))]
            # Add the mean average of the window
            smoothed.append(np.mean(bounded_values))
        except Exception as e:
            # If all values of the window are invalid (infinite or NaN), replace by 0.
            # Not the best, but not the worst way to do
            print(e)
            smoothed.append(0)
    return np.array(smoothed)


def remove_None_by_interpolation(values: Union[List, np.ndarray], extreme_values: float = 0) -> np.ndarray:
    """! Replace None values by interpolating with surrounding values
    If there is no not-None surrounding a point, it will use the "extreme_value" for the interpolation
    @attention This function is a very inefficient and should be replaced with a numpy/scipy function
    if you know a good one.
    @param values The series to modify
    @param extreme_values The value used for the lower bound or the higher bound if all previous or next values are
    None.

    @return the interpolated series
    """
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

            values[i] = value_after * (distance_before / (distance_before + distance_after)) + value_before * (
                    distance_after / (distance_before + distance_after))
    return values


def replace_trailing_and_leading_zeros_in_array(array: List, remove_leading: bool = True, remove_trailing: bool = True,
                                                replace_value: Any = None) -> np.ndarray:
    """! Replace the beginning or the ending of a series by a desired value if it begins or ends with 0's
    @param array The series to modify
    @param remove_leading Set to True to replace the 0's at he beginning with the "replace_value".
    @param remove_trailing Set to True to replace the 0's at he ending with the "replace_value".
    @param replace_value The value to use to replace the 0's.

    @return The modified version of the series.
    """
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


def visualize_data(data: Dict, time_in_seconds: bool = True, non_blocking: bool = False):
    """! Display a plot with the calculation of footsteps by distance and by velocity to compare the results
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param time_in_seconds Set to True to display the plots with time on the X-axis, otherwise it's the frame index
    @param non_blocking defines if the Pyplot window should interrupt the script at the end of the function
    """
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

    x_start = min(np.where(diff_left != None)[0][0], np.where(diff_right != None)[0][0],
                  np.where(feet_distance != None)[0][0])
    x_end = max(np.where(diff_left != None)[0][-1], np.where(diff_right != None)[0][-1],
                np.where(feet_distance != None)[0][-1])

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

    plt.show(block=not non_blocking)


def preprocess_data_for_walk_analysis(original_data):
    """! Process the data to remove None values and smooth the values to allow more
    precise computations for walk analysis
    @param original_data The skeleton data of the sequence coming from get_data_from_db()

    @return A smoothed version of the data coming from get_data_from_db()
    """
    data = copy.deepcopy(original_data)
    # Apply a smoothing process on the velocities
    for keypoint in data['velocities']:
        vel = np.array(data['velocities'][keypoint])
        vel = remove_None_by_interpolation(vel)
        vel = mean_smooth_data(vel, 2)
        vel = mean_smooth_data(vel, 2)
        data['velocities'][keypoint] = vel

    # Apply a smoothing process on the computed distances
    for key_couple in data['distances']:
        distance = np.array(data['distances'][key_couple])
        distance = mean_smooth_data(distance, 2)
        distance = mean_smooth_data(distance, 2)
        distance = truncate_values(distance, 3.00, convert_None_to=-0.00001)  # Consider distances >3m invalid
        data['distances'][key_couple] = distance

    return data


def reduce_walking_data_size(original_data: Dict, new_size: int) -> Dict:
    """! Reduce the size of the keypoint coords/velocities/distances dataset. This can be used to *maybe* create
    data augmentation on your dataset.
    @param original_data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param new_size The desired size of the reduced data

    @return A truncated version of the original walking sequence
    """
    data = copy.deepcopy(original_data)
    # Truncate the size of the arrays on each parameters of a keypoint
    for keypoint in data['velocities']:
        data['velocities'][keypoint] = data['velocities'][keypoint][:new_size]
        data['coordinates'][keypoint] = data['coordinates'][keypoint][:new_size]
        for keypoint2 in data['velocities']:
            data['distances'][f'{keypoint}-{keypoint2}'] = data['distances'][f'{keypoint}-{keypoint2}'][:new_size]
    data['frames'] = data['frames'][:new_size]
    data['timestamps'] = data['timestamps'][:new_size]

    return data


def compute_cycle_time(data: Dict, motion_threshold: float = 0.01) -> float:
    """! Get the Cycle time in seconds = duration of the sequence / number of footsteps
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param motion_threshold The threshold used (in m/s) to distinguish a static point from a moving one

    @return The cycle time extracted from the walking sequence
    """
    number_of_steps, times_of_steps = estimate_footsteps(preprocess_data_for_walk_analysis(data),
                                                         motion_threshold=motion_threshold)
    # Solution 1: Get the moment of the first and last step, and divide by the number of steps
    # Remove 1 step because the last timestamp is the beginning of a step, not a complete step
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


# TODO: Improve it by trimming before the first step / after last footstep
def compute_stance_ratio(data: Dict, motion_threshold: float = 0.01, concerned_foot: str = None) -> float:
    """! Compute the Stance ratio = time with two feet down / time with one foot moving
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param motion_threshold The threshold used (in m/s) to distinguish a static point from a moving one
    @param concerned_foot The foot for which you want to compute the stance ratio. Set to None if you want the
    mean value from the two feet (otherwise: 'LEFT' or 'RIGHT')

    @return The stance ratio extracted from a walking sequence
    """
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
        return float(
            np.mean([compute_stance_ratio(data=data, motion_threshold=motion_threshold, concerned_foot='right'),
                     compute_stance_ratio(data=data, motion_threshold=motion_threshold, concerned_foot='left')]))
    elif "left" in concerned_foot.lower():
        return np.sum(static_left) / np.sum(available_left)
    elif "right" in concerned_foot.lower():
        return np.sum(static_right) / np.sum(available_right)
    else:
        raise Exception("Invalid foot selected to compute stance ratio, use 'LEFT_ANKLE', 'RIGHT_ANKLE' or None")


# TODO: Improve it by trimming before the first step / after last footstep
def compute_stance_swing_ratio(data: Dict, motion_threshold: float = 0.01, concerned_foot: str = None) -> float:
    """! Compute the Stance/Swing ratio = time with two feet down / time with one foot moving
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param motion_threshold The threshold used (in m/s) to distinguish a static point from a moving one
    @param concerned_foot The foot for which you want to compute the stance/swing ratio. Set to None if you want the
    mean value from the two feet (otherwise: 'LEFT' or 'RIGHT')

    @return The stance/swing ratio extracted from a walking sequence
    """
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

# TODO: Improve it by trimming before the first step / after last footstep
def compute_double_support_time(data: Dict, motion_threshold: float = 0.01) -> float:
    """! Compute Double Support Time = time with the two feet on the ground
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param motion_threshold The threshold used (in m/s) to distinguish a static point from a moving one

    @return The estimated double support time (DST) extracted from a walking sequence
    """
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

def compute_walking_speed(data: Dict) -> float:
    """! Get the walking speed with the average of the movement of the hips
    @attention This function uses the average of the velocities recorded on the hips.
    Another way to estimate this with a higher accuracy when knowing that the person is walking in a linear path
    would be to compute the difference of coordinate at the beginning of the sequence and at the end of the sequence.
    This is a more generic but less accurate way to compute it.
    It is, still, a pretty good estimation.
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return The estimated walking speed
    """
    velocities_rh = np.array(data['velocities']['RIGHT_HIP'])
    velocities_rh = velocities_rh[np.logical_and(np.isfinite(velocities_rh), velocities_rh > 0)]
    velocities_lh = np.array(data['velocities']['LEFT_HIP'])
    velocities_lh = velocities_lh[np.logical_and(np.isfinite(velocities_lh), velocities_lh > 0)]
    return float(np.median([np.median(velocities_lh), np.median(velocities_rh)]))

def compute_stride_length(data: Dict) -> Tuple[float, float, float]:
    """! Estimate the length of strides from the skeleton of a walking sequence
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return A tuple containing the estimation of the stride's length, the estimation specific on the right foot
    and the one specific on the left foot.
    """
    # It is hard to decide if we want to use the footstep estimation by velocity or by distance,
    #   so we are using the two of them here
    frames = []
    # _, steps_instant = estimate_footsteps_by_distance(data, 3)
    # frames += steps_instant['frames'].tolist()
    _, steps_instant = estimate_footsteps(data, motion_threshold=0.01)
    frames += steps_instant['frames'].tolist()
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

    right_length = float(np.median(stride_lengths['right']))
    left_length = float(np.median(stride_lengths['left']))
    stride_length = float(np.mean([right_length, left_length]))
    return stride_length, right_length, left_length


#
# def compute_stride_length(data):
#     walking_speed = compute_walking_speed(data)
#     number_of_steps, times_of_steps = estimate_footsteps(preprocess_data_for_walk_analysis(data),
#                                                          motion_threshold=motion_threshold)
#     # Solution 1: Get the moment of the first and last step, and divide by the number of steps
#     # Remove 1 step because the last timestamp is the beginning of a step, not a complete step
#     sequence_duration = times_of_steps['timestamps'][-1] - times_of_steps['timestamps'][0]
#     total_distance_walked = sequence_duration * walking_speed
#     return total_distance_walked / (number_of_steps - 1)

def get_walking_direction(data: Dict) -> int:
    """! Indicates if a person is walking from North to South (1) or from South to North (-1) in the current (june 2021)
    experimental setup.
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return
    - -1 if walking from South to North,
    - 1 if walking from North to South,
    - 0 if the person doesn't seem to have been through the entire corridor
    """
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


def get_corridor_position(data: Dict) -> float:
    """! Estimate the distance of the person from the Eastern wall of the corridor
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return The distance in meters of the person from the eastern wall of the corridor in the current (june 2021)
    experimental setup
    """
    mean_pos = []
    percent_stripped = 0.0
    for i in range(len(data['frames'])):
        human = human_skeleton.HumanSkeleton().from_feet_analytics(data['coordinates'], i)
        if human.get_center_of_gravity() is not None:
            mean_pos.append(human.get_center_of_gravity())

    stripped_part = round(len(mean_pos) * percent_stripped)
    position = np.median(mean_pos[stripped_part: -max(1, stripped_part)], axis=0)

    # Estimate the corridor used by the X position
    return float(position[0])


def get_corridor_used(data: Dict) -> int:
    """! Estimate the part of the corridor used by the person when walking through the experimental setup
    @attention The limits used in the code were just estimated quickly and seems to misclassify a lot of
    sequences walking in the Western part (classified as Center). Maybe someone can modify the values of the limits.
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()

    @return
    - -1 if using the Eastern part,
    - 0 if using the Center part,
    - 1 if using the Western part
    """
    x_pos = get_corridor_position(data)
    # If it's on the West side, code it as 1
    if -2.5 <= x_pos <= -1.6:
        return 1
    # If it's in the center, code it as 0
    if -1.6 <= x_pos <= -1.0:
        return 0
    # If it's on the East side, code it as -1
    if -1.6 <= x_pos <= 0.0:
        return -1


def get_walking_features(data: Dict = None, id_sequence: int = None, time_between_sequences: float = 1.0,
                         server_side: bool = True, motion_threshold: float = 0.01, start_time: int = 0,
                         ignore_small_sequences: float = None) -> Dict:
    """! Compute all the features the functions of this script are able to extract on a sequence in database with
    the ID "id_sequence".
    @param data The skeleton data from the sequence after being transformed by preprocess_data_for_walk_analysis()
    @param id_sequence The ID of the sequence you want to analyse
    @param time_between_sequences Time between two timestamps of the database to consider it as two different sequences.
    @param server_side Set to True to access the database on the Listic server, set to False to access the local server
    @param motion_threshold The threshold used (in m/s) to distinguish a static point from a moving one
    @param start_time The date for which we consider having started the recordings of the experiments
    @param ignore_small_sequences Duration a sequence beneath which we can ignore

    @return A dictionary with all parameters:
    - nb_steps_motion
    - nb_steps_distance
    - cycle_time
    - cadence
    - stance_ratio_right
    - stance_ratio_left
    - double_support_ratio
    - footstep_width
    - walking_speed
    - stride_length
    - stride_length_right
    - stride_length_left
    - direction
    - corridor
    - corridor_position

    @see The functions called are :
    - estimate_footsteps()
    - estimate_footsteps_by_distance()
    - compute_cycle_time()
    - compute_stance_ratio()
    - compute_double_support_time()
    - estimate_footsteps_width()
    - compute_walking_speed()
    - compute_stride_length()
    - get_walking_direction()
    - get_corridor_used()
    - get_corridor_position()

    """
    if data is None:
        data = get_data_from_db(id_sequence, time_between_sequences=time_between_sequences, server_side=server_side,
                                start_time=start_time, ignore_small_sequences=ignore_small_sequences)
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
    corridor_pos = get_corridor_position(data)

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
              'corridor_position': corridor_pos,
              }
    return params


def display_walking_features(walking_features: Dict):
    """! Pretty display of a dictionary
    @param walking_features The dictionary you want to display
    """
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

    display_walking_features(get_walking_features(data=data))
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
