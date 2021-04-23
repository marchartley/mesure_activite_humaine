import json
import os
import pickle

import dateutil
import numpy as np
import pandas as pd

import human_skeleton
import glob

def get_all_speeds_recorded():
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
    pickle.dump(smoothed_sequences, open("all_tmp_datas_at_once.data", "wb+"))
    return smoothed_sequences


def split_sequences(pir_df: pd.DataFrame):
    all_sequences = []
    for seq_id in pir_df['sequence_id'].unique():
        all_sequences.append(pir_df[pir_df['sequence_id'] == seq_id])
    return all_sequences



pir_data = pd.read_csv("all_pir_values.csv")
sequences = split_sequences(pir_data)
# speeds = get_all_speeds_recorded()
speeds = {}
current_sequence_id = -1

all_data = {'coordinates': {}, 'timestamps': []}
print("fetching files")
for file in glob.glob("tmp_data*.json"):
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
    print(f"{round(((i+1)/len(sequences))*100)}%")
    start = dateutil.parser.isoparse(seq['time'].iloc[0].replace("Z", "+00:00")).timestamp() * 1000
    end = dateutil.parser.isoparse(seq['time'].iloc[-1].replace("Z", "+00:00")).timestamp() * 1000
    sequence_id = seq['sequence_id'].iloc[0]

    for frame_index in range(len(all_data['timestamps'])):
        human = human_skeleton.HumanSkeleton()
        human.from_feet_analitics(all_data['coordinates'], frame_index)
        if human.check_keypoints3D(["RIGHT_SHOULDER"]) or human.check_keypoints3D(["LEFT_SHOULDER"]) or human.check_keypoints3D(["RIGHT_HIP"]) or human.check_keypoints3D(["LEFT_HIP"]):
            if all_data['timestamps'][frame_index] < start or all_data['timestamps'][frame_index] > end:
                continue
            else:
                if sequence_id not in speeds:
                    speeds[sequence_id] = {'id': sequence_id,
                                           'start_pos': human.get_center_of_gravity(),
                                           'start_time': all_data['timestamps'][frame_index],
                                           'end_pos': None,
                                           'end_time': None}
                else:
                    speeds[sequence_id]['end_pos'] = human.get_center_of_gravity()
                    speeds[sequence_id]['end_time'] = all_data['timestamps'][frame_index]

print("100%")
print("Calculating speeds")
for seq_id in speeds:
    try:
        speeds[seq_id]['speed'] = np.linalg.norm(speeds[seq_id]['end_pos'] - speeds[seq_id]['start_pos']) / (speeds[seq_id]['end_time'] - speeds[seq_id]['start_time'])
        print(f"{speeds[seq_id]['id']} : {speeds[seq_id]['speed'] * 3.6}km/h")
    except Exception as e:
        print(e)