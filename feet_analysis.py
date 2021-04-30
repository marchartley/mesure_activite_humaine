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
    number_of_measurments = 0 # No use for this variable
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
            new_values.append(convert_None_to if values[i] is None else values[i] if values[i] < limit else new_values[-1])
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

# Display a plot with the calculation of footsteps by distance and by velocity
def visualize_data(data, time_in_seconds=True):
    footsteps_motion = estimate_footsteps(data, motion_threshold=0.01)[1]
    footsteps_distance = estimate_footsteps_by_distance(data, 3)[1]

    timestamps = np.array(data['timestamps'])
    frames = data['frames']
    timestamps = ((timestamps - timestamps[0]) / 1000)

    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0

    # Change the X-axis depending on the unit we want (frame number of time)
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
    plt.plot(x_axis, data['distances']['RIGHT_ANKLE-LEFT_ANKLE'], 'g-', label='Distance between feet')
    for foot_estimation in motion_detection_distance:
        plt.axvline(x=foot_estimation, color=(.0, .0, 1.0, .4))
    plt.ylabel('Distance (m)')
    plt.title("Distance between feet in time")
    plt.xlabel(x_label)
    plt.legend()

    plt.show()

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


# Get the Cycle time = number of footsteps / duration of the sequence
# TODO: Improve it by trimming before the first step / after last footstep
def compute_cycle_time(data, motion_threshold=0.01):
    number_of_steps = estimate_footsteps(data, motion_threshold=motion_threshold)[0]
    sequence_duration = (data['timestamps'][-1] - data['timestamps'][0]) / 1000
    return number_of_steps / sequence_duration

# Compute the Stance/Swing ratio = time with two feet down / time with one foot moving
# TODO: Improve it by trimming before the first step / after last footstep
def compute_stance_swing_ratio(data, motion_threshold=0.01, concerned_foot=None):
    diff_right = data['velocities']['RIGHT_ANKLE'] - data['velocities']['LEFT_ANKLE']
    diff_left = data['velocities']['LEFT_ANKLE'] - data['velocities']['RIGHT_ANKLE']
    diff_right[diff_right < 0] = 0
    diff_left[diff_left < 0] = 0
    if concerned_foot is None:
        if len(diff_right[diff_right <= motion_threshold]) == 0 or len(diff_left[diff_left <= motion_threshold]) == 0:
            return 0
        return ((len(diff_right) / len(diff_right[diff_right <= motion_threshold])) +
                (len(diff_left) / len(diff_left[diff_right <= motion_threshold]))) / 2
    elif "left" in concerned_foot.lower():
        if len(diff_left[diff_left <= motion_threshold]) == 0:
            return 0
        return len(diff_left) / len(diff_left[diff_left <= motion_threshold])
    elif "right" in concerned_foot.lower():
        if len(diff_right[diff_right <= motion_threshold]) == 0:
            return 0
        return len(diff_right) / len(diff_right[diff_right <= motion_threshold])
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
    return len(diff_right[abs(diff_right - diff_left) < motion_threshold] < motion_threshold) / len(diff_right)

# Get the walking speed with the average of the movement of the shoulders/hips
# This function should maybe not be used...
def compute_walking_speed(data):
    velocities_rs = np.array(data['velocities']['RIGHT_SHOULDER'])
    velocities_rs = velocities_rs[np.logical_and(np.isfinite(velocities_rs), velocities_rs > 0)]
    velocities_ls = np.array(data['velocities']['LEFT_SHOULDER'])
    velocities_ls = velocities_ls[np.logical_and(np.isfinite(velocities_ls), velocities_ls > 0)]
    velocities_rh = np.array(data['velocities']['RIGHT_HIP'])
    velocities_rh = velocities_rh[np.logical_and(np.isfinite(velocities_rh), velocities_rh > 0)]
    velocities_lh = np.array(data['velocities']['LEFT_HIP'])
    velocities_lh = velocities_lh[np.logical_and(np.isfinite(velocities_lh), velocities_lh > 0)]
    return np.mean([np.mean(velocities_lh), np.mean(velocities_rh),
                    np.mean(velocities_rs), np.mean(velocities_ls)])


if __name__ == "__main__":
    motion_threshold = 0.01  # 0.01 m/s ~= 0.5m/min
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
    # Compute all the parameters by changing the data size to get the evolution in time
    for i in range(2, len(data['timestamps'])):  # From all data, to only 2 frames
        data = reduce_walking_data_size(original_data, i)
        footsteps_per_velo.append(estimate_footsteps(data, motion_threshold=motion_threshold)[0])
        footsteps_per_dist.append(estimate_footsteps_by_distance(data)[0])
        cycle_times.append(compute_cycle_time(data, motion_threshold=motion_threshold))
        stance_swings['left'].append(
            compute_stance_swing_ratio(data, concerned_foot='left', motion_threshold=motion_threshold))
        stance_swings['right'].append(
            compute_stance_swing_ratio(data, concerned_foot='right', motion_threshold=motion_threshold))
        stance_swings['mean'].append(compute_stance_swing_ratio(data, motion_threshold=motion_threshold))
        double_supports.append(compute_double_support_time(data, motion_threshold=motion_threshold))
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
