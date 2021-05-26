import datetime
import os
import pickle
import time

import dateutil

import InfluxDB_connection as db
import feet_analysis
import pandas as pd

create_CSV_from_scratch = False

# Name of the output file
output_filename = 'gait_values.csv'

start_time_process = time.time()

# Create DB client
client = db.ListicClientDB(server_side=True)

start_time = None
starting_sequence_id = 0
old_data = None
if create_CSV_from_scratch or not os.path.exists(output_filename):
    # Start the sequences from 26/04/2021 because before data were incorrect
    start_time = 1619395200000  # 1621342821000  #
else:
    old_data = pd.read_csv(output_filename, index_col=0)
    start_time = (dateutil.parser.isoparse(old_data.iloc[-1]['time']).timestamp()) * 1000
    starting_sequence_id = old_data.iloc[-1]['sequence_id'] + 1


# Get all walking sequences (split sequences if there is at least 1 sec between them)
print('Segmenting sequences from Database...')
time_between_sequences = 1.0
sequences_boundaries = client.get_splitted_sequences_boundaries(time_between_sequences=time_between_sequences,
                                                                start_time=start_time, ignore_small_sequences=0.2)
walking_sequences = client.get_skeletons_in_splitted_sequences(time_between_sequences=time_between_sequences,
                                                               start_time = start_time, ignore_small_sequences=0.2)

fetching_errors = []
print("Fetching data and computing gait features...")
sequence_data = []
# Get the PIR values for each sequence
for id_sequence in range(len(walking_sequences)):
    print(f'{round(100 * id_sequence / len(walking_sequences), 1)}% ... Sequence {id_sequence + starting_sequence_id} '
          f'({datetime.datetime.fromtimestamp(sequences_boundaries[id_sequence][0]/1000).strftime("%d/%m %H:%M:%S")} - '
          f'{datetime.datetime.fromtimestamp(sequences_boundaries[id_sequence][1]/1000).strftime("%H:%M:%S")}) ',
          end = "")
    # If there are very few data in this sequence, don't bother work on it
    try:
        if max([len(walking_sequences[id_sequence][0][0][key]) for key in walking_sequences[id_sequence][0][0]]) < 5:
            print(f'ignored, not enough skeleton data')
            continue
        print(end = ".")
    except KeyError as e:
        print(f"ignored, skeleton data wrongly saved")
        continue

    try:
        print(end=".")
        # Get PIR values for this sequence
        pir_values = client.get_pir_values_from_walking_sequence(start_time=sequences_boundaries[id_sequence][0],
                                                                 end_time=sequences_boundaries[id_sequence][1],
                                                                 time_offset=5.0)
        print(end=".")
        if len(pir_values['time']) == 0:
            print(f"ignored, no PIR values")
            continue
        print(end=".")
        # Compute gait features for this specific sequence
        gait_features = feet_analysis.get_walking_features(time_between_sequences = time_between_sequences,
                                                           id_sequence=id_sequence,
                                                           server_side=True, motion_threshold=0.8,
                                                           start_time = start_time, ignore_small_sequences=0.2)
        print(end=".")
        # Store these values in an array
        sequence_data.append({
            'id': id_sequence + starting_sequence_id,
            'PIR': pir_values,
            'gait': gait_features
        })
        print(f"OK")
    except Exception as e:
        fetching_errors.append({'id_sequence': id_sequence + starting_sequence_id, 'error': e})
        if isinstance(e, KeyError):
            print(f"ignored, possibly that this keypoint has not been found in the sequence: {e}")
        else:
            print(f"ignored, other error: {e}")
print('100%')

print('Reformatting data...')
# Transform all these values into a DataFrame
full_df = None

dataframe_errors = []
if old_data is not None:
    full_df = old_data
for sequence in sequence_data:
    try:
        temp_df = pd.DataFrame(sequence['PIR'])
        number_of_data = temp_df.shape[0]
        for feature in sequence['gait']:
            # Add this value to the entire column
            temp_df[feature] = sequence['gait'][feature]
        temp_df['sequence_id'] = sequence['id']
        if full_df is None:
            full_df = temp_df
            # print(full_df)
        else:
            full_df = full_df.append(temp_df, ignore_index=True)
            # print(full_df)
    except Exception as e:
        dataframe_errors.append({'sequence': sequence, 'error': e})

print(f"Saving file to '{os.path.abspath(output_filename)}', {len(sequence_data)} sequences saved")
full_df.to_csv(output_filename)

pickle.dump({'fetch': fetching_errors, 'dataframe': dataframe_errors}, open('error_log.log', 'wb+'))
if len(fetching_errors) > 0:
    print(f'Encountered {len(fetching_errors)} errors while fetching:')
    for err in fetching_errors:
        print(f'\t- seq#{err["id_sequence"]}: {str(err["error"])}')

if len(dataframe_errors) > 0:
    print(f'Encountered {len(dataframe_errors)} errors while transforming:')
    for err in dataframe_errors:
        print(f'\t- seq#{err["sequence"]["id"]}: {str(err["error"])}')
