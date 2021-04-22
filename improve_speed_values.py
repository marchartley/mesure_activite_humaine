import json

import numpy as np

import human_skeleton
import glob

def get_all_speeds_recorded():
    time_cooldown = 100
    sequences = []
    times = []
    for file in glob.glob('tmp_data*.json'):
        data = json.load(open(file, 'r'))

        cooldown = time_cooldown
        sequence_starting = False
        sequence_started_index = 0
        starting_human = None
        previous_human = None
        last_time_index = 0

        start_time = 0 # data['timestamps'][0]
        if data['timestamps'][-1]*1000 < 1619010000000:
            continue
        for frame_index in range(len(data['timestamps'])):
            human = human_skeleton.HumanSkeleton()
            human.from_feet_analitics(data['coordinates'], frame_index)

            if human.check_keypoints3D(['LEFT_SHOULDER']) or human.check_keypoints3D(['RIGHT_SHOULDER']) or human.check_keypoints3D(['LEFT_HIP']) or human.check_keypoints3D(['RIGHT_HIP']) \
                    and frame_index < len(data['timestamps']) -1:
                if not sequence_starting:
                    sequence_starting = True
                    sequence_started_index = frame_index
                    starting_human = human
                previous_human = human
                last_time_index = frame_index
                cooldown = time_cooldown
            else:
                if starting_human != previous_human:
                    if sequence_starting and (cooldown < 0 or frame_index == len(data['timestamps']) -1):
                        # cooldown = time_cooldown
                        displacement = np.linalg.norm(previous_human.get_center_of_gravity() - starting_human.get_center_of_gravity())
                        displacement /= ((data['timestamps'][last_time_index]/1000) - (data['timestamps'][sequence_started_index]/1000))
                        sequences.append(displacement)
                        sequence_starting = False
                        times.append((data['timestamps'][sequence_started_index], data['timestamps'][last_time_index]))
                    else:
                        cooldown -= 1

    sequences = np.array(sequences)
    # print(sequences)
    smoothed_sequences = []
    for i in range(len(sequences)):
        smoothed_speed = 0
        if i == 0:
            smoothed_speed = sequences[i]*.8 + sequences[i + 1]*.2
        elif i == len(sequences) - 1:
            smoothed_speed = sequences[i - 1]*.2 + sequences[i]*.8
        else:
            smoothed_speed = sequences[i - 1]*.1 + sequences[i]*.8 + sequences[i + 1]*.1

        smoothed_sequences.append([times[i], smoothed_speed])

    smoothed_sequences.sort(key = lambda seq: seq[0][0])
    return smoothed_sequences
