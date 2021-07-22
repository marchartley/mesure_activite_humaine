"""! This script creates a CSV file containing sensors data when somebody is walking through them.
The walking gait characteristics are computed and stored with the sensors data
"""

import datetime
import os
import pickle
import dateutil
import pandas as pd

import InfluxDB_connection as db
import deep_neural_training
import feet_analysis

from utils import HiddenPrints


def fill_csv(create_CSV_from_scratch: bool = False,
             output_filename: str = 'gait_values.csv',
             client: db.ListicClientDB = None,
             start_time=None, end_time=None,
             verbose: bool = True,
             use_cache: bool = True,
             sensor_height: str = None) -> pd.DataFrame:
    """! Creates the CSV file that stores the sensors data and the ground truth from the camera
    @param create_CSV_from_scratch If true, the previous CSV file is deleted and the calculations are all done again
    @param output_filename name of the CSV file to create
    @param client InfluxDB client to access the database
    @param start_time Timestamp of the time at which we limit our calculations
    @param end_time Timestamp of the time at which we limit our calculations
    @param verbose Display logs
    @param use_cache Use the InfluxDB client to use the cache in order to avoid computations when segmenting
    the sequences
    @param sensor_height The height of the sensors (the tag 'sensor_height' in InfluxDB). If None, all sequences are
    used

    @return All the data from the sensors and the features computed from the camera, as a DataFrame
    """
    with HiddenPrints(verbose):
        if client is None:
            # Create DB client
            client = db.ListicClientDB(server_side=True)

        starting_sequence_id = 0
        old_data = None
        if create_CSV_from_scratch or not os.path.exists(output_filename):
            # Start the sequences from 26/04/2021 because before data were incorrect
            if start_time is None:
                start_time = 1619395200000  # 1621342821000  #
        else:
            old_data = pd.read_csv(output_filename, index_col=0)
            start_time = (dateutil.parser.isoparse(old_data.iloc[-1]['time']).timestamp()) * 1000
            starting_sequence_id = old_data.iloc[-1]['sequence_id'] + 1

        # Get all walking sequences (split sequences if there is at least 3 sec between them)
        print('Segmenting sequences from Database...')
        time_between_sequences = 3.0
        if not use_cache:
            client.get_splitted_sequences_boundaries.cache_clear()
            client.get_skeletons_in_splitted_sequences.cache_clear()

        sequences_boundaries = client.get_splitted_sequences_boundaries(time_between_sequences=time_between_sequences,
                                                                        start_time=start_time, end_time=end_time,
                                                                        ignore_small_sequences=0.2)
        walking_sequences = client.get_skeletons_in_splitted_sequences(time_between_sequences=time_between_sequences,
                                                                       start_time=start_time, end_time=end_time,
                                                                       ignore_small_sequences=0.2)

        fetching_errors = []
        print("Fetching data and computing gait features...")
        sequence_data = []
        # Get the PIR values for each sequence
        for id_sequence in range(len(walking_sequences)):
            print(
                f'{round(100 * id_sequence / len(walking_sequences), 1)}% ... Sequence {id_sequence + starting_sequence_id} '
                f'({datetime.datetime.fromtimestamp(sequences_boundaries[id_sequence][0] / 1000).strftime("%d/%m %H:%M:%S")} - '
                f'{datetime.datetime.fromtimestamp(sequences_boundaries[id_sequence][1] / 1000).strftime("%H:%M:%S")}) ',
                end="", flush=True)
            # If there are very few data in this sequence, don't bother work on it
            try:
                if max([len(walking_sequences[id_sequence][0][0][key]) for key in
                        walking_sequences[id_sequence][0][0]]) < 5:
                    print(f'ignored, not enough skeleton data')
                    continue
                print(end=".", flush=True)
            except KeyError as e:
                print(f"ignored, skeleton data wrongly saved")
                continue

            try:
                print(end=".", flush=True)
                # Get PIR values for this sequence
                pir_values = client.get_pir_values_from_walking_sequence(
                    start_time=sequences_boundaries[id_sequence][0],
                    end_time=sequences_boundaries[id_sequence][1],
                    time_offset=15.0, sensor_height=sensor_height)
                print(end=".")
                if len(pir_values['time']) == 0:
                    print(f"ignored, no PIR values")
                    continue

                print(end=".", flush=True)
                # Compute gait features for this specific sequence
                gait_features = feet_analysis.get_walking_features(time_between_sequences=time_between_sequences,
                                                                   id_sequence=id_sequence,
                                                                   server_side=True, motion_threshold=0.8,
                                                                   start_time=start_time, ignore_small_sequences=0.2)
                if gait_features['direction'] == 0:
                    print(f"ignored, direction undefined")
                    continue

                speed = gait_features['walking_speed']
                before_offset = 0
                after_offset = 0
                # Going from South to North, we are missing ~4m at the end of the recording and 1m before
                if gait_features['direction'] == -1:
                    before_offset = 3 / speed
                    after_offset = 5 / speed
                # Going from North to South, we are missing ~1m at the end of the recording and 4m before
                else:
                    before_offset = 3 / speed
                    after_offset = 5 / speed

                # We have data with 10s before and after, so we remove (10 - x) sec of data
                # PIR values are recorded at 40Hz, so we remove (10 - x) * 40 lines of data
                before_offset = int(max(1, 10 - before_offset) * 40)
                after_offset = int(max(1, 10 - after_offset) * 40)

                for measure in pir_values:
                    pir_values[measure] = pir_values[measure][before_offset:(-after_offset)]

                print(end=".", flush=True)
                # Store these values in an array
                sequence_data.append({
                    'id': id_sequence + starting_sequence_id,
                    'PIR': pir_values,
                    'gait': gait_features
                })
                print(f"OK ({round(len(pir_values['pir0']) / 40)}s)")
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
                # raise e
                dataframe_errors.append({'sequence': sequence, 'error': e})

        if len(sequence_data) > 0:
            print(f"Saving file to '{os.path.abspath(output_filename)}', {len(sequence_data)} sequences saved")
            full_df.to_csv(output_filename)
        else:
            print('No new data to save')

        pickle.dump({'fetch': fetching_errors, 'dataframe': dataframe_errors}, open('error_log.log', 'wb+'))
        if len(fetching_errors) > 0:
            print(f'Encountered {len(fetching_errors)} errors while fetching:')
            for err in fetching_errors:
                print(f'\t- seq#{err["id_sequence"]}: {str(err["error"])}')

        if len(dataframe_errors) > 0:
            print(f'Encountered {len(dataframe_errors)} errors while transforming:')
            for err in dataframe_errors:
                print(f'\t- seq#{err["sequence"]["id"]}: {str(err["error"])}')

    return pd.read_csv(output_filename)

def main():
    """! The creation of the CSV file and the file containing the Wavelet Transforms

    @note Modify directly the code of this file if you want to change the sensor's height criteria
    """
    sensors_height = 'ankles'
    pir_data = fill_csv(create_CSV_from_scratch=False,
                        output_filename=f"gait_values_{sensors_height}.csv",
                        verbose=True,
                        start_time=datetime.datetime(year=2021, month=6, day=4).timestamp() * 1000,
                        sensor_height=sensors_height)
    deep_neural_training.precompute_morlet_transforms(pir_data, f"all_pirs_transformed_{sensors_height}.data", output_filename=True)


if __name__ == '__main__':
    main()
