"""! This script defines a client that will connect your program to the InfluxDB database in order
to do request and do basic operations
"""


import random
import time

import influxdb as db
import numpy as np
import recording
import datetime
import math
from functools import lru_cache
import dateutil.parser
import pandas as pd
from typing import Union, List, Dict, Tuple


class ListicClientDB:
    """! The InfluxDB client that can be used in our programs"""
    def __init__(self, server_side: bool = None,
                 host: str = None, port: int = None, login: str = None, pwd: str = None, database: str = None):
        """! Constructor
        @param server_side Setting this to True will configure the client to access the real server from Listic
        without having to define the other parameters. Setting it to False will set the other parameters to
        be able to access directly your local server (if correctly installed).
        If trying to connect to Listic server, you have to be connected on the WifiListic (or have access to
        the address 192.168.143.29)
        @param host The address to connect to the server.
        @param port The port number on which your InfluxDB instance is installed (default: 8086)
        @param login The login used when you connect to InfluxDB (same as Grafana)
        @param pwd The password used when you connect to InfluxDB (same as Grafana)
        @param database The database you want to use in InfluxDB
        """
        ## The values that needs to be injected in InfluxDB are stored temporarily in this array
        self.stored_points = []
        ## Some walking features can be estimated and stored in the database, but
        # it is not used after (if I remember well)
        self.walking_metrics = []
        ## The real instance of the InfluxDB client
        self.client = None

        if server_side:
            if host is None:
                host = '192.168.143.29'
            if port is None:
                port = 8086
            if login is None:
                login = 'listic'
            if pwd is None:
                pwd = 'groupefiad'
            if database is None:
                database = 'iot'
        else:
            if host is None:
                host = 'localhost'
            if port is None:
                port = 8086
            if login is None:
                login = 'admin'
            if pwd is None:
                pwd = 'admin'
            if database is None:
                database = 'iot'

        # Create the client and connect directly to the good database
        self.client = db.InfluxDBClient(host, port, login, pwd)
        self.client.switch_database(database)
        try:
            # Try to ping the server to be sure it is well configured
            self.client.ping()
        except Exception as e:
            print("The DB is unreachable...")
            raise e

    def insert_skeleton_keypoints(self, coord2D: Dict = None, coord3D: Dict = None, timestamp: float = None,
                                  skeleton_id: int = 0,
                                  device: str = None, ignore_None_values: bool = True, send_immediately: bool = True,
                                  walking_speed: float = None, footsteps_width: float = None,
                                  number_of_footsteps: float = None, stride_length: float = None,
                                  step_length: float = None, timestamp_unit: str = 'ms', raw_coord_2D: Dict = None,
                                  raw_coord_3D: Dict = None) -> None:
        """! Insert coordinates of a skeleton by passing the array like the one that is created by the
        HumanSkeleton class
        @see HumanSkeleton

        @param coord2D The coordinates on the screen that you want to save
        @param coord3D The coordinates on the real world that you want to save
        @param timestamp The timestamp at which these coordinates has been recorded
        @param skeleton_id The ID of the person in the sequence. Most processes takes only the sequences with the
        person ID == 0, so try to have all skeleton on a sequence with a unique person to have an ID = 0.
        @param device The name of the device that recorded the person. You then are able to filter sequences filmed
        by a camera D400 or the L515, or any new camera you are using.
        @param ignore_None_values Set to False if you want to insert a NULL value for keypoints that aren't seen on
        screen. I'm not really sure about the changes needed if you insert NULL values, so the best is to keep this
        argument to True, I guess...
        @param send_immediately Set to True to send these coordinates to InfluxDB (which can take up to 0.5 second)
        or set it to False to wait to have more data and send all the data at once.
        @param walking_speed If already computed, you can store the estimation of the walking speed.
        @param footsteps_width If already computed, you can store the estimation of the footsteps width.
        @param number_of_footsteps If already computed, you can store the estimation of the number of footsteps.
        @param stride_length If already computed, you can store the estimation of the stride length.
        @param step_length If already computed, you can store the estimation of the step length.
        @param timestamp_unit The unit used as timestamp ('ms' or 's')
        @param raw_coord_2D The coordinates without any translation/rotation, in case the transformation had an math
        error
        @param raw_coord_3D The coordinates without any translation/rotation, in case the transformation had an math
        error
        """
        if self.client is None:
            raise Exception(f"Client of {self.__class__.__name__} wasn't configure correctly")

        # If coord2D is given, save each keypoint's coordinate
        if coord2D is not None:
            for i, key in enumerate(coord2D):
                if ignore_None_values or np.all(coord2D[key]) is not None:
                    self.stored_points.append({
                        'measurement': 'skeleton_detection_2d',
                        'time': round(timestamp),
                        'tags': {
                            'device': device,
                            'skeleton_id': skeleton_id,
                            'keypoint': key
                        },
                        'fields': {
                            'x': coord2D[key][0],
                            'y': coord2D[key][1],
                        }
                    })
        # If raw_coord_2D is given, save each keypoint's coordinate
        if raw_coord_2D is not None:
            for i, key in enumerate(raw_coord_2D):
                if ignore_None_values or np.all(raw_coord_2D[key]) is not None:
                    self.stored_points.append({
                        'measurement': 'skeleton_detection_2d_raw',
                        'time': round(timestamp),
                        'tags': {
                            'device': device,
                            'skeleton_id': skeleton_id,
                            'keypoint': key
                        },
                        'fields': {
                            'x': raw_coord_2D[key][0],
                            'y': raw_coord_2D[key][1],
                        }
                    })

        # If coord3D is given, save each keypoint's coordinate
        if coord3D is not None:
            for i, key in enumerate(coord3D):
                if ignore_None_values or np.all(coord3D[key]) is not None:
                    self.stored_points.append({
                        'measurement': 'skeleton_detection_3d',
                        'time': round(timestamp),
                        'tags': {
                            'device': device,
                            'skeleton_id': skeleton_id,
                            'keypoint': key
                        },
                        'fields': {
                            'x': coord3D[key][0],
                            'y': coord3D[key][1],
                            'z': coord3D[key][2],
                        }
                    })
        # If raw_coord_3D is given, save each keypoint's coordinate
        if raw_coord_3D is not None:
            for i, key in enumerate(raw_coord_3D):
                if ignore_None_values or np.all(raw_coord_3D[key]) is not None:
                    self.stored_points.append({
                        'measurement': 'skeleton_detection_3d_raw',
                        'time': round(timestamp),
                        'tags': {
                            'device': device,
                            'skeleton_id': skeleton_id,
                            'keypoint': key
                        },
                        'fields': {
                            'x': raw_coord_3D[key][0],
                            'y': raw_coord_3D[key][1],
                            'z': raw_coord_3D[key][2],
                        }
                    })

        # Also insert the walking parameters if given
        self.insert_walking_metrics(timestamp, walking_speed,
                                    footsteps_width, number_of_footsteps,
                                    stride_length, step_length, send_immediately, timestamp_unit,
                                    skeleton_id, device, ignore_None_values)

        if send_immediately:
            # If asked, send the data to InfluxDB
            self.send_to_db(timestamp_unit=timestamp_unit)

    def send_to_db(self, timestamp_unit='ms'):
        """! Function that send all the data stored by this client into the InfluxDB server
        @note Note
        This operation can take some time (up to 0.5s), so avoid doing this too many times and think about setting
        "send_immediately" argument from "insert_skeleton_keypoints()" to False if it's not needed
        @see insert_skeleton_keypoints()

        @param timestamp_unit The unit used on your timestamps. If your data hasn't been saved at the desired timestamp,
        check the units.
        """
        self.client.write_points(self.stored_points, time_precision=timestamp_unit)
        self.client.write_points(self.walking_metrics, time_precision=timestamp_unit)
        self.stored_points = []
        self.walking_metrics = []

    def insert_walking_metrics(self, timestamp: float, walking_speed: float,
                               footsteps_width: float, number_of_footsteps: float,
                               stride_length: float, step_length: float, send_immediately: bool = False,
                               timestamp_unit: str = 'ms', skeleton_id: int = 0,
                               device: str = None, ignore_None_values: bool = True) -> None:
        """! Function that insert the computed walking parameters into the database.
        @note Note:
        The data in this measurement are never used after in this project, so I'm not taking the time to describe it.
        """
        if self.client is None:
            raise Exception(f"Client of {self.__class__.__name__} wasn't configure correctly")
        additional_fields = {}
        if walking_speed is not None and walking_speed is not np.nan and walking_speed > 0:
            additional_fields['walking_speed'] = float(walking_speed)
        if footsteps_width is not None and footsteps_width is not np.nan and footsteps_width > 0:
            additional_fields['footsteps_width'] = float(footsteps_width)
        if number_of_footsteps is not None and number_of_footsteps is not np.nan and number_of_footsteps > 0:
            additional_fields['number_of_footsteps'] = float(number_of_footsteps)
        if stride_length is not None and stride_length is not np.nan and stride_length > 0:
            additional_fields['stride_length'] = float(stride_length)
        if step_length is not None and step_length is not np.nan and step_length > 0:
            additional_fields['step_length'] = float(step_length)

        if len(additional_fields) > 0:
            pir_timestamp = self.get_closest_timestamp(timestamp)
            self.walking_metrics.append({
                'measurement': 'walking_metrics',
                'time': round(pir_timestamp),
                'tags': {},
                'fields': additional_fields,
            })
        if send_immediately:
            self.send_to_db(timestamp_unit=timestamp_unit)

    @lru_cache(maxsize=1)
    def get_all_skeletons(self) -> Tuple[Dict[int, dict], Dict[int, dict]]:
        """! Function to retrieve skeletons stored since 26/04/2021.
        @deprecated This function is never really used because it is time consuming. Use other functions
        like get_skeletons_in_splitted_sequences() to work with skeletons.
        @return A tuple containing the coordinates as a Dict with data in 2D and another Dict for coordinates 3D.
        """
        return self.retrieve_skeleton_keypoints(start_time=1619395200000)  # = 26/04/2021 00:00
        # return self.retrieve_skeleton_keypoints(start_time=1619395200000, end_time=1619422738000)  #= 26/04/2021 00:00

    @lru_cache(maxsize=1)
    def get_splitted_presence_or_not_boundaries(self, start_time: int = None, end_time: int = None,
                                                time_between_sequences: float = 2.0,
                                                ignore_small_sequences: float = None) \
            -> Dict[str, List[Tuple[int, int]]]:
        """! Function to get the temporal boundaries of sequences during which a person is walking and sequences
        where nobody is walking. This can be used to create a dataset for a classifier that do person detection and
        maybe (with improvement in this code, I guess) an estimator of number of persons.

        @param start_time The lower limit at which we can detect a sequence (in ms)
        @param end_time The upper limit at which we can detect a sequence (in ms)
        @param time_between_sequences The needed time (in sec) between two timestamps for which we consider that
        doesn't belong in the same sequence.
        @param ignore_small_sequences If some skeletons are detected in a sequence but that this sequence is smaller
        than a threshold (in sec), ignore it.

        @return A dictionary with keys:
        - 'presence': an array of tuple (start_time, end_time) of sequences with someone walking
        - 'no_presence': an array of tuple(start_time, end_time) of sequences without anybody

        @see If you don't care for absence sequences, see get_splitted_sequences_boundaries().
        """
        # Get the temporal boundaries of persons walking, then we'll be able to compute the sequences where
        # we are sure there was nobody.
        user_presence = self.get_splitted_sequences_boundaries(start_time=start_time, end_time=end_time,
                                                               time_between_sequences=time_between_sequences,
                                                               ignore_small_sequences=ignore_small_sequences)
        # The duration of "no presence" will get the same distribution, with an extra 5sec
        users_durations = np.array([t[1] - t[0] + 5000 for t in user_presence])
        no_user_durations = abs(np.random.normal(users_durations.mean(), users_durations.std(),
                                                 len(users_durations) - 1))

        # Store in an array the available timeslots with (available time, start, end, mean timestamp)
        # Keep at least 5 sec between a presence and an absence
        no_user_available_durations = [[user_presence[seq_index][0] - user_presence[seq_index - 1][1] - 5000 * 2,
                                        user_presence[seq_index - 1][1],
                                        user_presence[seq_index][0],
                                        (user_presence[seq_index][0] + user_presence[seq_index - 1][1]) / 2]
                                       for seq_index in range(1, len(user_presence))]

        no_user_available_durations = list(filter(lambda x: x[0] > 0 and
                                                            len(self.pir_values_to_json(
                                                                self.get_all_pir_values(
                                                                    start=x[1] + 5000,
                                                                    end=x[2] - 5000,
                                                                    group_by_move=False))['time']
                                                                ) > 35 * x[0] / 1000, no_user_available_durations))

        # Save the repartition in a JSON {index of the available timeslot: [list of the index of the wished durations]}
        repartition_of_no_user = {}
        for id_duration in range(len(no_user_durations)):
            for id_time_slot in range(len(no_user_available_durations)):
                if no_user_available_durations[id_time_slot][0] > no_user_durations[id_duration]:
                    no_user_available_durations[id_time_slot][0] -= no_user_durations[id_duration]
                    if id_time_slot not in repartition_of_no_user:
                        repartition_of_no_user[id_time_slot] = []
                    repartition_of_no_user[id_time_slot].append(id_duration)
                    break

        boundaries_for_no_user = []
        # Compute the timestamps for each non-user slots
        for id_time_slot in repartition_of_no_user:
            real_duration = np.array([no_user_durations[x] for x in repartition_of_no_user[id_time_slot]]).sum()
            current_start = no_user_available_durations[id_time_slot][3] - real_duration / 2
            for id_duration in repartition_of_no_user[id_time_slot]:
                boundaries_for_no_user.append((current_start, current_start + no_user_durations[id_duration]))
                current_start += no_user_durations[id_duration]

        return {'presence': user_presence, 'no_presence': boundaries_for_no_user}

    @lru_cache(maxsize=1)
    def get_splitted_sequences_boundaries(self, sequence_filename: str = None, start_time: int = None,
                                          end_time: int = None, time_between_sequences: float = 20.0,
                                          ignore_small_sequences: float = None) -> List[Tuple[int, int]]:
        """! Function to get the temporal boundaries of sequences during which a person is walking in front of sensors.
        This function can be useless as so, but is used to improve execution speed
        of get_skeletons_in_splitted_sequences().

        @param sequence_filename If given a rosbag file (recording from the camera), the start and end
        limits are extracted from it. But rosbag files are not really used anymore...
        @param start_time The lower limit at which we can detect a sequence (in ms)
        @param end_time The upper limit at which we can detect a sequence (in ms)
        @param time_between_sequences The needed time (in sec) between two timestamps for which we consider that
        doesn't belong in the same sequence.
        @param ignore_small_sequences If some skeletons are detected in a sequence but that this sequence is smaller
        than a threshold (in sec), ignore it.

        @return A dictionary with keys:
        - 'presence': an array of tuple (start_time, end_time) of sequences with someone walking
        - 'no_presence': an array of tuple(start_time, end_time) of sequences without anybody
        """
        # If you provide a rosbag file and don't set the start/end limits, the limits are computed automatically
        if start_time is None and end_time is None:
            if sequence_filename is None:
                pass
            else:
                pipeline = recording.start_playback(sequence_filename, realtime=False)
                start_time = recording.get_frame_timestamp(pipeline.wait_for_frames())
                duration = pipeline.get_active_profile().get_device().as_playback().get_duration()
                pipeline = recording.stop_playback(pipeline)
                duration = (duration / datetime.timedelta(microseconds=1)) / 1e3
                end_time = start_time + duration

        # First, create the temporal limit conditions for the InfluxDB query
        time_conditions = []
        if start_time:
            start_time = math.floor(start_time)
            time_conditions.append(str(start_time) + "ms <= time")
        if end_time:
            end_time = math.ceil(end_time)
            time_conditions.append("time <= " + str(end_time) + "ms")
        condition = ""
        if len(time_conditions) > 0:
            condition = "WHERE " + " and ".join(time_conditions)

        # Get the data from InfluxDB, not looking at the keypoint.
        query = f'SELECT "x" FROM "autogen"."skeleton_detection_3d" {condition}'
        results = self.client.query(query)

        # Split all the data to group them into clusters of time and find the first and last timestamp of each cluster
        sequences_interval = []
        for key, points in results.items():
            # Get the timestamps in seconds
            timestamps = np.array(sorted(list(set(
                [dateutil.parser.isoparse(point['time'].replace("Z", "+00:00")).timestamp() for point in points]))))
            # Split these values to get groups of timestamps that are closer than "time_between_sequences" seconds apart
            sequences_interval = list(map(
                lambda arr: (arr[0] * 1000, arr[-1] * 1000),
                np.split(timestamps, np.where(np.diff(timestamps) > time_between_sequences)[0] + 1)))

        # Remove sequences with a duration too low
        if ignore_small_sequences is not None and ignore_small_sequences is not False and ignore_small_sequences > 0:
            sequences_interval = list(filter(lambda seq: seq[1] - seq[0] > ignore_small_sequences * 1000,
                                             sequences_interval))
        return sequences_interval

    @lru_cache(maxsize=1)
    def get_skeletons_in_splitted_sequences(self, sequence_filename: str = None, start_time: int = None,
                                            end_time: int = None, merge_skeleton_ids: bool = True,
                                            time_between_sequences: float = 20.0,
                                            sequence_index: int = None, ignore_small_sequences: float = None)\
            -> Union[List[Tuple], Tuple]:
        """! Function to get the skeletons of persons walking in front of the camera divided in sequences.
        @param sequence_filename If given a rosbag file (recording from the camera), the start and end
        limits are extracted from it. But rosbag files are not really used anymore...
        @param start_time The lower limit at which we can detect a sequence (in ms)
        @param end_time The upper limit at which we can detect a sequence (in ms)
        @param merge_skeleton_ids Useless
        @param time_between_sequences The needed time (in sec) between two timestamps for which we consider that
        doesn't belong in the same sequence.
        @param sequence_index Set this value to the ID of a sequence to get only this specific sequence.
        -1 to get the last sequence.
        @param ignore_small_sequences If some skeletons are detected in a sequence but that this sequence is smaller
        than a threshold (in sec), ignore it.

        @return If "sequence_index" is defined, return a tuple containing a dictionary that defines all coordinates
        of a sequence with coordinates in 2D and coordinates in 3D as defined by get_splitted_sequences_boundaries(). \n
        Otherwise, return an array of these tuple for each sequence found.
        """
        sequences_interval = self.get_splitted_sequences_boundaries(sequence_filename, start_time, end_time,
                                                                    time_between_sequences,
                                                                    ignore_small_sequences)
        # Apply the normal skeleton retrieval with the time boundaries found before
        if sequence_index is not None:
            return self.retrieve_skeleton_keypoints(start_time=sequences_interval[sequence_index][0],
                                                    end_time=sequences_interval[sequence_index][1])
        skeletons = []
        for interval in sequences_interval:
            skeletons.append(self.retrieve_skeleton_keypoints(start_time=interval[0], end_time=interval[1]))
        return skeletons

    def get_pir_values_from_walking_sequence(self, skeleton_data: Dict = None, start_time: int = None,
                                             end_time: int = None, id_sequence: int = None, time_offset: float = 5.0,
                                             time_between_sequences: float = 20.0,
                                             sensor_height: str = None) -> Dict[str, List[int]]:
        """! From the skeleton data extracted during a sequence (by get_splitted_sequences_boundaries() or
        get_skeletons_in_splitted_sequences() ) or by defining temporal boundaries, extract the data
        from sensors during this sequence.
        @param skeleton_data The skeleton data extracted from a sequence with the functions
        get_splitted_sequences_boundaries() or get_skeletons_in_splitted_sequences().
        @param start_time If the skeleton data is not provided, give the starting time (in ms) of a sequence to extract
        the sensors data.
        @param end_time If the skeleton data is not provided, give the ending time (in ms) of a sequence to extract the
        sensors data.
        @param id_sequence If none of the above is provided, just give the ID of the sequence and it will fetch the
        good starting/ending time. (But I'm pretty sure this method is not the best possible...)
        @param time_offset The time (in sec) that is added before and after the camera stopped seen somebody. In our
        experiments, the sensors had a greater field of view so it is important to get some offset time.
        @param time_between_sequences If using the "id_sequence", it's the needed time (in sec) between two
        timestamps for which we consider that doesn't belong in the same sequence.
        @param sensor_height Providing this parameter let the system filter to get only sequences where the sensors
        were in a certain position. For now the choice is "hips" or "ankles". But maybe we will add "one_sided", or even
        "other_sensors_hips", "other_sensors_ankles", etc...
        """
        pir_start = None
        pir_end = None
        # If we have the skeleton data, the minimal/maximal timestamp is stored in the dictionary
        if skeleton_data is not None:
            min_time = min([skeleton_data[keypoint][0]['time_ms'] for keypoint in skeleton_data])
            max_time = max([skeleton_data[keypoint][-1]['time_ms'] for keypoint in skeleton_data])
            pir_start = min_time
            pir_end = max_time
        # If we have the start and end, just use them
        elif start_time is not None and end_time is not None:
            pir_start = start_time
            pir_end = end_time
        # If we have the id_sequence, we split the sequences and extract only the interesting one
        elif id_sequence is not None:
            boundaries = self.get_splitted_sequences_boundaries(time_between_sequences=time_between_sequences,
                                                                ignore_small_sequences=0.2)
            pir_start, pir_end = boundaries[id_sequence]
        else:
            raise Exception("One of those parameters must be used to get the PIR sequence:\n"
                            "- skeleton_data (dictionary coming from "
                            "InfluxDB_connection.retrieve_skeleton_keypoints() )\n"
                            "- start_time and end_time\n"
                            "- sequence_id and time_between_sequences")
        # Just add the additional time offset
        pir_start -= time_offset * 1000
        pir_end += time_offset * 1000

        # Extract the PIR values and convert them into a JSON/dictionary variable
        return self.pir_values_to_json(self.get_all_pir_values(start=pir_start, end=pir_end,
                                                               group_by_move=False, sensor_height=sensor_height))

    def pir_values_to_json(self, query_results: db.client.ResultSet) -> Dict[str, List[int]]:
        """! From the results of a query to InfluxDB, transform the data into a JSON/dictionary variable
        @param query_results The result of a query from the client with self.client.query()

        @return A dictionary containing all the data from each sensor and all the timestamps from the data
        """
        pirs = {
            'time': [],
            'pir0': [],
            'pir1': [],
            'pir2': [],
            'pir3': [],
        }
        for key, val in query_results.items():
            for values in val:
                pirs['time'].append(values['time'])
                pirs['pir0'].append(values['pir0'])
                pirs['pir1'].append(values['pir1'])
                pirs['pir2'].append(values['pir2'])
                pirs['pir3'].append(values['pir3'])
        return pirs

    def pir_values_to_DF(self, query_results: db.client.ResultSet) -> pd.DataFrame:
        """! From the results of a query to InfluxDB, transform the data into a DataFrame
        @param query_results The result of a query from the client with self.client.query()

        @return A DataFrame containing all the data from each sensor and all the timestamps from the data
        """
        return pd.DataFrame(self.pir_values_to_json(query_results)).astype(int)

    # @lru_cache(maxsize=1)
    # def get_all_precomputed_speeds(self):
    #     query = f'SELECT "walking_speed", "footsteps_width", "number_of_footsteps", "stride_length", "step_length" ' \
    #             f'FROM "walking_metrics" WHERE time > \'2021-04-21 00:00:00\''
    #     results = self.client.query(query)
    #     return results

    def timestamp_to_date_format(self, timestamp: float, unit: str = 'ms'):
        """! Small function used to transform a timestamp to a pretty date format (for visual purpose)
        It can also be used to format the queries sent to InfluxDB.
        """
        if unit == 's':
            timestamp /= 1000
        date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        return date.strftime('%Y-%m-%dT%H:%M:%SZ')

    @lru_cache(maxsize=2)
    def get_all_pir_values(self, start: Union[str, float] = None, end: Union[str, float] = None,
                           group_by_move: bool = False, sensor_height: str = None):
        """! Extract all the data from the sensors database
        @param start The lower time limit for which we extract the data (as a formatted string or as a timestamp in ms)
        @param end  The upper time limit for which we extract the data (as a formatted string or as a timestamp in ms)
        @param group_by_move Set to True to separate the extracted data depending on the predefined moving direction.
        This was useful when we were not taking data "in the wild", now we don't have any predefined moves
        @param sensor_height Providing this parameter let the system filter to get only sequences where the sensors
        were in a certain position. For now the choice is "hips" or "ankles". But maybe we will add "one_sided", or even
        "other_sensors_hips", "other_sensors_ankles", etc...
        :return:
        """
        if start is None:
            start = '2021-04-21 00:00:00'

        if not isinstance(start, str):
            start = self.timestamp_to_date_format(start, unit='s')
        where_condition = f'WHERE time > \'{start}\''
        if end is not None:
            if not isinstance(end, str):
                end = self.timestamp_to_date_format(end, unit='s')
            where_condition += f' AND time < \'{end}\''

        if sensor_height is not None:
            where_condition += f" AND sensor_height = '{sensor_height}'"

        group_by_condition = 'GROUP BY move' if group_by_move else ''

        query = f'SELECT "pir0", "pir1", "pir2", "pir3" FROM "pir_network" {where_condition} {group_by_condition}'
        results = self.client.query(query)
        return results

    @lru_cache(maxsize=2)
    def get_all_pir_timestamps(self):
        """! Function to extract all the timestamps at which the sensors were in use.
        @deprecated This function was in use quite a while ago, I don't see any use of this anymore.
        (used in insert_walking_metrics(), which is a useless function...)
        """
        all_values = []
        for key, values in self.get_all_pir_values(group_by_move=True).items():
            for data in values:
                all_values.append(dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000)
        return all_values

    def get_closest_timestamp(self, timestamp: float):
        """! Function to find the closest data from the sensors relative to the timestamp given as input.
        @deprecated This function was in use quite a while ago, I don't see any use of this anymore.
        (used in insert_walking_metrics(), which is a useless function...)
        @param timestamp the timestamp you want to compare with.
        """
        closest_timestamp = 0
        min_delta = math.inf
        for current_timestamp in self.get_all_pir_timestamps():
            if abs(current_timestamp - timestamp) < min_delta:
                min_delta = abs(current_timestamp - timestamp)
                closest_timestamp = current_timestamp
        return closest_timestamp

    def retrieve_skeleton_keypoints(self, sequence_filename: str = None, start_time: float = None,
                                    end_time: float = None, merge_skeleton_ids: bool = True):
        """! Function to get the skeletons of persons walking in front of the camera.
        This function is the one used by get_skeletons_in_splitted_sequences().
        @param sequence_filename If given a rosbag file (recording from the camera), the start and end
        limits are extracted from it. But rosbag files are not really used anymore...
        @param start_time The lower limit at which we can detect a sequence (in ms)
        @param end_time The upper limit at which we can detect a sequence (in ms)
        @param merge_skeleton_ids If True, skeleton coordinates are extracted as if only one person was walking.
        Otherwise, the results will contain one list of coordinates per person.

        @return A tuple containing a dictionary that defines all coordinates
        of a sequence with coordinates in 2D and coordinates in 3D
        """
        # If no start/end is defined, it is automatically computed from a rosbag file provided
        if start_time is None and end_time is None:
            if sequence_filename is None:
                pass
            else:
                pipeline = recording.start_playback(sequence_filename, realtime=False)
                start_time = recording.get_frame_timestamp(pipeline.wait_for_frames())
                duration = pipeline.get_active_profile().get_device().as_playback().get_duration()
                pipeline = recording.stop_playback(pipeline)
                duration = (duration / datetime.timedelta(microseconds=1)) / 1e3
                end_time = start_time + duration

        # Create the WHERE for the query
        time_conditions = []
        if start_time:
            start_time = math.floor(start_time)
            time_conditions.append(str(start_time) + "ms <= time")
        if end_time:
            end_time = math.ceil(end_time)
            time_conditions.append("time <= " + str(end_time) + "ms")
        condition = ""
        if len(time_conditions) > 0:
            condition = "WHERE " + " and ".join(time_conditions)

        # Create the GROUP BY condition for the query (by keypoints and/or by skeleton_id)
        groupby = []
        if not merge_skeleton_ids:
            groupby.append('"skeleton_id"')
        groupby.append('"keypoint"')
        groupby_condition = 'GROUP BY ' + ', '.join(groupby)

        # Get data as 2D coordinates (screen coordinates) or as 3D (real-world with Aruco box as reference)
        query2D = f'SELECT "x", "y" FROM "autogen"."skeleton_detection_2d" {condition} {groupby_condition}'
        query3D = f'SELECT "x", "y", "z" FROM "autogen"."skeleton_detection_3d" {condition} {groupby_condition}'
        # print(query2D)

        results2D = self.client.query(query2D)
        points2D = {}
        results3D = self.client.query(query3D)
        points3D = {}

        # Extract data from the query results object into a Dict
        for key, point in results2D.items():
            skeleton_id = 0 if merge_skeleton_ids else int(key[1]['skeleton_id'])
            if skeleton_id not in points2D:
                points2D[skeleton_id] = {}
            keypoint = key[1]['keypoint']
            points2D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
                lambda coord:
                dict(coord,
                     **{'time_ms': dateutil.parser.isoparse(coord['time'].replace("Z", "+00:00")).timestamp()}),
                point)))

        for key, point in results3D.items():
            skeleton_id = 0 if merge_skeleton_ids else int(key[1]['skeleton_id'])
            if skeleton_id not in points3D:
                points3D[skeleton_id] = {}
            keypoint = key[1]['keypoint']
            points3D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
                lambda coord:
                dict(coord,
                     **{'time_ms': dateutil.parser.isoparse(coord['time'].replace("Z", "+00:00")).timestamp()}),
                point)))

        return points2D, points3D

    @staticmethod
    def check_if_body_will_reappear(skeleton_data: dict, current_timestamp: float) -> bool:
        """! Function that tells us if there are still somebody detected in a sequence after a given time
        @param skeleton_data The data given by retrieve_skeleton_keypoints() that defines the skeleton in a sequence
        @param current_timestamp The time for which we want to check if the person will be back in screen

        @return A boolean telling if the person will be back on screen after the given timestamp.
        """
        max_timestamp = 0
        for keypoint in skeleton_data:
            max_timestamp = max(max_timestamp, max([data['time_ms'] for data in skeleton_data[keypoint]]))
            if max_timestamp > current_timestamp:
                return True
        # If the person doesn't reappear, return false
        return False


def main():
    import matplotlib.pyplot as plt

    client = ListicClientDB(server_side=True)

    for i in range(1, 10):
        res = client.get_pir_values_from_walking_sequence(id_sequence=-i, time_between_sequences=1.0)
        pirs = client.pir_values_to_DF(res)
        pirs.plot()
        plt.show()


if __name__ == '__main__':
    main()
