import json
import os
import pickle

import dateutil
import numpy as np
import pandas as pd

import InfluxDB_connection
import feet_analysis
import human_skeleton
import glob

# Some values are going to be very low or very high because of the camera out-of-sight problem
speed_high_bound = 3.0
speed_low_bound = 0.1


def get_all_gait_sequences():
    client = InfluxDB_connection.ListicClientDB(server_side=True)

    motion_threshold = 0.8  # 0.8m/s ~ 3km/h
    sequences_boundaries = client.get_splitted_sequences_boundaries(time_between_sequences=1.0)
    sequences_parameters = []

    for sequence_id in range(len(sequences_boundaries)):
        try:
            data = feet_analysis.get_data_from_db(sequence_id, 1.0)
            footsteps_motion = feet_analysis.estimate_footsteps(data, motion_threshold=motion_threshold)[0]
            footsteps_distance = feet_analysis.estimate_footsteps_by_distance(data, 3)[0]
            cycle_time = feet_analysis.compute_cycle_time(data)
            cadence = 60 / cycle_time
            stance_ratio_right = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                                                                    concerned_foot="RIGHT_ANKLE")
            stance_ratio_left = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                                                                   concerned_foot="LEFT_ANKLE")
            double_support_ratio = feet_analysis.compute_double_support_time(data, motion_threshold=motion_threshold)
            footstep_width = feet_analysis.estimate_footsteps_width(data)
            walking_speed = feet_analysis.compute_walking_speed(data)
            stride_length, right_length, left_length = feet_analysis.compute_stride_length(data)
            params = {'start': sequences_boundaries[sequence_id][0],
                      'end': sequences_boundaries[sequence_id][1],
                      'nb_steps_motion': footsteps_motion,
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
                      'stride_length_left': left_length
                      }
            sequences_parameters.append(params)
        except:
            pass
    json.dump(sequences_parameters, open('sequence_parameters.json', 'w+'))


def get_all_speeds_recorded():
    # if os.path.exists("all_tmp_datas_at_once.data"):
    #     return pickle.load(open("all_tmp_datas_at_once.data", "rb"))

    cooldown = 500  # time in ms between two "mini-sequences", data frequency ~20Hz
    max_sequence_length = 10
    sequences = []
    times = []

    client = InfluxDB_connection.ListicClientDB(server_side=True)
    _, skeletons3D = client.get_all_skeletons()
    skeletons3D = skeletons3D[0]

    sequence_starting = False
    sequence_started_index = 0
    starting_human = None
    previous_human = None
    last_time_index = 0

    # We will focus on a unique keypoint to compute the speed
    keypoint_for_speed_computing = 'LEFT_SHOULDER'

    # Get all the timestamps recorded
    timestamps = [observation['time_ms'] for observation in skeletons3D[keypoint_for_speed_computing]]

    frames_with_keypoint = 0
    # For each "frame"
    for i_frame in range(max_sequence_length, len(timestamps), 2):
        previous_index = i_frame - max_sequence_length

        if timestamps[i_frame] - timestamps[previous_index] < cooldown / 1000:
            # Compute the displacement and deduce the speed
            starting_coord = np.array([skeletons3D[keypoint_for_speed_computing][previous_index]['x'],
                                       skeletons3D[keypoint_for_speed_computing][previous_index]['y'],
                                       skeletons3D[keypoint_for_speed_computing][previous_index]['z']])
            ending_coord = np.array([skeletons3D[keypoint_for_speed_computing][i_frame]['x'],
                                     skeletons3D[keypoint_for_speed_computing][i_frame]['y'],
                                     skeletons3D[keypoint_for_speed_computing][i_frame]['z']])

            displacement = np.linalg.norm(ending_coord - starting_coord)
            displacement /= (timestamps[i_frame] - timestamps[previous_index])

            if speed_low_bound <= displacement <= speed_high_bound:
                sequences.append(displacement)
                # It's the end of the mini-sequence
                sequence_starting = False
                times.append((timestamps[previous_index] * 1000, timestamps[i_frame] * 1000))

    sequences = np.array(sequences)
    # print(sequences)
    smoothed_sequences = list(zip(times, sequences))
    # for i in range(len(sequences)):
    #     smoothed_speed = sequences[i]
    #     smoothed_sequences.append([times[i], smoothed_speed])

    # Sort the array on its starting timestamp
    smoothed_sequences.sort(key=lambda sequence: sequence[0][0])
    pickle.dump(smoothed_sequences, open("all_tmp_datas_at_once.data", "wb+"))
    return smoothed_sequences


def get_all_speeds_recorded_local():
    if os.path.exists("all_tmp_datas_at_once.data"):
        return pickle.load(open("all_tmp_datas_at_once.data", "rb"))
    time_cooldown = 100
    sequences = []
    times = []
    for file in glob.glob('tmp_data*.json'):
        data = json.load(open(file, 'rb'))

        cooldown = time_cooldown
        sequence_starting = False
        sequence_started_index = 0
        starting_human = None
        previous_human = None
        last_time_index = 0

        start_time = 0  # data['timestamps'][0]

        for frame_index in range(len(data['timestamps'])):
            human = human_skeleton.HumanSkeleton()
            human.from_feet_analytics(data['coordinates'], frame_index)
            if human.check_keypoints3D('LEFT_SHOULDER') and frame_index < len(data['timestamps']) - 1:
                if not sequence_starting:
                    sequence_starting = True
                    sequence_started_index = frame_index
                    starting_human = human
                previous_human = human
                last_time_index = frame_index
                cooldown = time_cooldown
            else:
                if starting_human != previous_human:
                    if sequence_starting and (cooldown < 0 or frame_index == len(data['timestamps']) - 1):
                        # cooldown = time_cooldown
                        displacement = np.linalg.norm(
                            np.array(previous_human.get_keypoint3D('LEFT_SHOULDER')) - np.array(
                                starting_human.get_keypoint3D('LEFT_SHOULDER')))
                        displacement /= ((data['timestamps'][last_time_index] / 1000) - (
                                    data['timestamps'][sequence_started_index] / 1000))
                        sequences.append(displacement)
                        sequence_starting = False
                        times.append((data['timestamps'][sequence_started_index], data['timestamps'][last_time_index]))
                    else:
                        cooldown -= 1

    sequences = np.array(sequences)
    # print(sequences)
    smoothed_sequences = []
    for i in range(len(sequences)):
        smoothed_speed = sequences[i]
        # if i == 0:
        #     smoothed_speed = sequences[i]*.8 + sequences[i + 1]*.2
        # elif i == len(sequences) - 1:
        #     smoothed_speed = sequences[i - 1]*.2 + sequences[i]*.8
        # else:
        #     smoothed_speed = sequences[i - 1]*.1 + sequences[i]*.8 + sequences[i + 1]*.1

        smoothed_sequences.append([times[i], smoothed_speed])

    # Sort the array on its starting timestamp
    smoothed_sequences.sort(key=lambda sequence: sequence[0][0])
    pickle.dump(smoothed_sequences, open("all_tmp_datas_at_once.data", "wb+"))
    return smoothed_sequences


def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(pir_df[pir_df['sequence_id'] == seq_id])
    return all_sequences


if __name__ == '__main__':
    get_all_gait_sequences()
    print("All data saved as sequence_parameters.json!")
    exit()
    pir_data = pd.read_csv("all_pir_values.csv")
    sequences = split_sequences(pir_data)
    # speeds = get_all_speeds_recorded()
    speeds = {}
    current_sequence_id = -1

    all_data = {'coordinates': {}, 'timestamps': []}
    print("fetching files")
    for file in glob.glob("tmp_data_*.json"):
        data = json.load(open(file, "r"))
        for key in data['coordinates']:
            if key not in all_data['coordinates']:
                all_data['coordinates'][key] = []
            all_data['coordinates'][key].extend(data['coordinates'][key])
        all_data['timestamps'].extend(data['timestamps'])

    print("Sorting")
    for key in all_data['coordinates']:
        all_data['coordinates'][key] = [coord for _, coord in zip(all_data['timestamps'], all_data['coordinates'][key])]
    all_data['timestamps'] = sorted(all_data['timestamps'])

    print("Parsing")
    for i, seq in enumerate(sequences):
        print(f"{round(((i + 1) / len(sequences)) * 100)}%")
        start = dateutil.parser.isoparse(seq['time'].iloc[0].replace("Z", "+00:00")).timestamp() * 1000
        end = dateutil.parser.isoparse(seq['time'].iloc[-1].replace("Z", "+00:00")).timestamp() * 1000
        sequence_id = seq['sequence_id'].iloc[0]

        for frame_index in range(len(all_data['timestamps'])):
            human = human_skeleton.HumanSkeleton()
            human.from_feet_analytics(all_data['coordinates'], frame_index)
            number_of_keys = 0
            for key in ['LEFT_SHOULDER']:  # , 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']:
                if human.check_keypoints3D(key):
                    number_of_keys += 10
            if number_of_keys > 1:
                if all_data['timestamps'][frame_index] < start or all_data['timestamps'][frame_index] > end:
                    continue
                else:
                    if sequence_id not in speeds:
                        speeds[sequence_id] = {'id': sequence_id,
                                               'start_pos': human.get_keypoint3D('LEFT_SHOULDER'),
                                               'start_time': all_data['timestamps'][frame_index],
                                               'end_pos': None,
                                               'end_time': None}
                    else:
                        speeds[sequence_id]['end_pos'] = human.get_keypoint3D('LEFT_SHOULDER')
                        speeds[sequence_id]['end_time'] = all_data['timestamps'][frame_index]

    print("100%")
    print("Calculating speeds")
    for seq_id in speeds:
        try:
            speeds[seq_id]['speed'] = np.linalg.norm(
                np.array(speeds[seq_id]['end_pos']) - np.array(speeds[seq_id]['start_pos'])) / (
                                                  speeds[seq_id]['end_time'] - speeds[seq_id]['start_time'])
            print(
                f"{speeds[seq_id]['id']} : {round(speeds[seq_id]['speed'] * 1000, 2)}m/s ({speeds[seq_id]['start_time']} - {speeds[seq_id]['end_time']})")
        except Exception as e:
            print(e)
