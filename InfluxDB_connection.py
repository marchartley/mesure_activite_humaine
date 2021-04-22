import influxdb as db
import numpy as np
import recording
import datetime
import math
from functools import lru_cache
import dateutil.parser


class ListicClientDB():
    def __init__(self, server_side=None, host=None, port=None, login=None, pwd=None, database=None):
        self.stored_points = []
        self.walking_metrics = []
        self.walking_metrics_to_pir = []
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

        self.client = db.InfluxDBClient(host, port, login, pwd)
        self.client.switch_database(database)
        try:
            self.client.ping()
        except Exception as e:
            print("The DB is unreachable...")
            raise e

    def insert_skeleton_keypoints(self, coord2D=None, coord3D=None, timestamp=None, skeleton_id=0,
                                  device=None, ignore_None_values=True, send_immediatly=True,
                                  walking_speed=None, footsteps_width=None, number_of_footsteps=None,
                                  stride_length=None, step_length=None,
                                  timestamp_unit='ms'):
        if self.client is None:
            raise Exception(f"Client of {self.__class__.__name__} wasn't configure correctly")

        additionnal_fields = {}
        if walking_speed is not None and walking_speed is not np.nan and walking_speed > 0:
            additionnal_fields['walking_speed'] = float(walking_speed)
        if footsteps_width is not None and footsteps_width is not np.nan and footsteps_width > 0:
            additionnal_fields['footsteps_width'] = float(footsteps_width)
        if number_of_footsteps is not None and number_of_footsteps is not np.nan and number_of_footsteps > 0:
            additionnal_fields['number_of_footsteps'] = float(number_of_footsteps)
        if stride_length is not None and stride_length is not np.nan and stride_length > 0:
            additionnal_fields['stride_length'] = float(stride_length)
        if step_length is not None and step_length is not np.nan and step_length > 0:
            additionnal_fields['step_length'] = float(step_length)

        if coord2D is not None:
            for i, key in enumerate(coord2D):
                if ignore_None_values and np.all(coord2D[key]) is not None:
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
        if coord3D is not None:
            for i, key in enumerate(coord3D):
                if ignore_None_values and np.all(coord3D[key]) is not None:
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
        if len(additionnal_fields) > 0:
            # print(additionnal_fields)
            self.walking_metrics.append({
                'measurement': 'walking_metrics',
                'time': round(timestamp),
                'tags': {
                    'device': device,
                    'skeleton_id': skeleton_id,
                },
                'fields': additionnal_fields,
            })
            self.insert_walking_metrics_to_pir_network(timestamp, walking_speed,
                                                       footsteps_width, number_of_footsteps,
                                                       stride_length, step_length)

        if send_immediatly:
            self.send_to_db(timestamp_unit=timestamp_unit)

    def send_to_db(self, timestamp_unit='ms'):
        self.client.write_points(self.stored_points, time_precision=timestamp_unit)
        self.client.write_points(self.walking_metrics, time_precision=timestamp_unit)
        self.client.write_points(self.walking_metrics_to_pir, time_precision=timestamp_unit)
        self.stored_points = []
        self.walking_metrics = []
        self.walking_metrics_to_pir = []

    def insert_walking_metrics_to_pir_network(self, timestamp, walking_speed,
                                              footsteps_width, number_of_footsteps,
                                              stride_length, step_length):
        pir_timestamp = self.get_closest_timestamp(timestamp)
        additionnal_fields = {}
        if walking_speed is not None and walking_speed is not np.nan and walking_speed > 0:
            additionnal_fields['walking_speed'] = float(walking_speed)
        if footsteps_width is not None and footsteps_width is not np.nan and footsteps_width > 0:
            additionnal_fields['footsteps_width'] = float(footsteps_width)
        if number_of_footsteps is not None and number_of_footsteps is not np.nan and number_of_footsteps > 0:
            additionnal_fields['number_of_footsteps'] = float(number_of_footsteps)
        if stride_length is not None and stride_length is not np.nan and stride_length > 0:
            additionnal_fields['stride_length'] = float(stride_length)
        if step_length is not None and step_length is not np.nan and step_length > 0:
            additionnal_fields['step_length'] = float(step_length)

        if len(additionnal_fields) > 0:
            self.walking_metrics_to_pir.append({
                'measurement': 'pir_network',
                'time': round(pir_timestamp),
                'tags': {},
                'fields': additionnal_fields,
            })
            # print(pir_timestamp)

    #            self.client.write_points(self.stored_points, time_precision=timestamp_unit)

    @lru_cache(maxsize=2)
    def get_all_pir_values(self):
        # query = f'SELECT "pir0", "pir1", "pir2", "pir3" FROM "autogen"."pir_network"'
        query = f'SELECT "pir0", "pir1", "pir2", "pir3", "walking_speed" FROM "pir_network" WHERE time > \'2021-04-21 12:00:00\' GROUP BY move'
        results = self.client.query(query)
        return results

    @lru_cache(maxsize=2)
    def get_all_pir_timestamps(self):
        all_values = []
        for key, values in self.get_all_pir_values().items():
            for data in values:
                all_values.append(dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000)
        return all_values

    def get_closest_timestamp(self, timestamp):
        closest_timestamp = 0
        min_delta = math.inf
        for current_timestamp in self.get_all_pir_timestamps():
            if abs(current_timestamp - timestamp) < min_delta:
                min_delta = abs(current_timestamp - timestamp)
                closest_timestamp = current_timestamp
        return closest_timestamp

    def retrieve_skeleton_keypoints(self, sequence_filename=None, start_time=None, end_time=None):
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

        query2D = f'SELECT "x", "y" FROM "autogen"."skeleton_detection_2d" {condition} GROUP BY "skeleton_id", "keypoint"'
        query3D = f'SELECT "x", "y", "z" FROM "autogen"."skeleton_detection_3d" {condition} GROUP BY "skeleton_id", "keypoint"'

        results2D = self.client.query(query2D)
        points2D = {}
        for key, point in results2D.items():
            skeleton_id = int(key[1]['skeleton_id'])
            if skeleton_id not in points2D:
                points2D[skeleton_id] = {}
            keypoint = key[1]['keypoint']
            points2D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
                lambda coord:
                dict(coord,
                     **{'time_ms': datetime.datetime.fromisoformat(coord['time'].replace('Z', '+00:00')).timestamp()}),
                point)))

        results3D = self.client.query(query3D)
        points3D = {}
        for key, point in results3D.items():
            skeleton_id = int(key[1]['skeleton_id'])
            if skeleton_id not in points3D:
                points3D[skeleton_id] = {}
            keypoint = key[1]['keypoint']
            points3D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
                lambda coord:
                dict(coord,
                     **{'time_ms': datetime.datetime.fromisoformat(coord['time'].replace('Z', '+00:00')).timestamp()}),
                point)))

        return points2D, points3D

    @staticmethod
    def check_if_body_will_reapear(skeleton_data: dict, current_timestamp: float):
        max_timestamp = 0
        for keypoint in skeleton_data:
            max_timestamp = max(max_timestamp, max([data['time_ms'] for data in skeleton_data[keypoint]]))

        return max_timestamp > current_timestamp
