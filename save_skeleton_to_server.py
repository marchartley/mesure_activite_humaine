import os

import cv2

import feet_analysis
import recording
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time
import json

import influxdb as db

import matplotlib.pyplot as plt

import convert_realsense_to_real_world as coord_converter

import human_skeleton

import InfluxDB_connection
import datetime
import glob
import math
from cv2 import aruco

import realsense_filters as filters


def save_data(bag_filename, client=None, skeletrack=None, device='realsense1', use_aruco_coord=True, compute_from_tmp_data=False):
    if client is None:
        client = InfluxDB_connection.ListicClientDB(server_side=False)  # Connect locally if no InfluxDB client is given
        print("No specific InfluxDB client is given, continuing with local client")

    coordinates = {}
    distances = {}
    velocities = {}
    timestamps = []
    frame_indexes = []
    median_center = [0, 0, 0]
    median_x = [1, 0, 0]
    median_y = [0, 1, 0]
    median_z = [0, 0, 1]
    motion_threshold = 0.01

    if not compute_from_tmp_data:
        num_frame = 0

        start_time = time.time()
        human = None

        current_timestamp = 0
        previous_timestamp = 0

        number_of_frames = 0

        start_timestamp = None

        if use_aruco_coord:
            print("Calibrating camera...")
        pipeline = recording.start_playback(bag_filename, pipeline=rs.pipeline(), realtime=False)

        if skeletrack is None:
            skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline=pipeline, image_is_upside_down=True)
        else:
            skeletrack.set_pipeline(pipeline)

        while True:
            try:
                frames = pipeline.wait_for_frames(200)
                number_of_frames += 1
                current_timestamp = recording.get_frame_timestamp(frames)
                timestamp = current_timestamp
                if start_timestamp is None:
                    start_timestamp = timestamp
                if previous_timestamp == current_timestamp:
                    continue
            except Exception as e:
                print(e)
                break

            try:
                color_frame = np.array(frames.get_color_frame().get_data())
                depth_frame = frames.get_depth_frame()
                # depth_frame = filters.l515_filter(depth_frame).as_depth_frame()
            except Exception as e:
                print(f"Error on the file {bag_filename}:", e)
                recording.stop_playback(pipeline)
                return

            try:
                median_center, median_x, median_y, median_z = coord_converter.calibrate_aruco_cube(color_frame,
                                                                                                   aruco.Dictionary_get(
                                                                                                       aruco.DICT_6X6_250),
                                                                                                   depth_frame,
                                                                                                   pipeline)

                skeletons = skeletrack.track_skeletons(color_frame, debug_time=False)

                has_human_presence = False
                if skeletrack.check_user_presence(color_image=color_frame):
                    human = human_skeleton.HumanSkeleton()
                    try:
                        human.from_tracker(skeletrack, 0, depth_frame)

                        coord2D, coord3D = human.to_json()
                        has_human_presence = True
                    except:
                        pass

                for keypoint in map(lambda keypoint: keypoint.name, human_skeleton.BodyParts):
                    if keypoint not in coordinates:
                        coordinates[keypoint] = []

                    # Coord not found
                    if not has_human_presence or keypoint not in coord2D or np.any(coord2D[keypoint] == -1) or np.any(
                            coord2D[keypoint] is None):
                        coordinates[keypoint].append(np.array([None, None, None]))
                        continue
                    else:  # If coord is found
                        coordinates[keypoint].append(coord3D[keypoint])

                timestamps.append(current_timestamp)
                frame_indexes.append(num_frame)

            except Exception as e:
                print(f"Error on the file {bag_filename}, it might have been recorded with USB2.0 or USB1.0...")
                recording.stop_playback(pipeline)
                raise e
                return
        try:
            print("Stoping...")
            pipeline.stop()
        except:
            pass
        print("Video parsed")
        # Compute velocities
        for keypoint in coordinates:
            previous_pos = np.array(coordinates[keypoint][:-1])
            next_pos = np.array(coordinates[keypoint][1:])  # Shift the array by one
            number_of_velocities = len(coordinates[keypoint]) - 1
            movement = np.array([0] * number_of_velocities)
            for i in range(number_of_velocities):
                # if keypoint == "RIGHT_ANKLE":
                # 	print(previous_pos[i], next_pos[i])
                if None in previous_pos[i] or None in next_pos[i]:
                    continue
                try:
                    movement[i] = (np.linalg.norm(next_pos[i] - previous_pos[i]) / (timestamps[i] - timestamps[i - 1])) * 1000
                except:
                    # print(np.linalg.norm(next_pos[i] - previous_pos[i]), timestamps[i] - timestamps[i - 1], timestamps[i], timestamps[i-1])
                    continue
            velocities[keypoint] = np.append([0], movement).tolist()
        # print(velocities[keypoint])

        max_range = 0
        for key in coordinates:
            max_range = max(max_range, len(coordinates[key]))
        # Compute distances
        for keypoint1 in human_skeleton.BodyParts:
            key1 = keypoint1.name
            for keypoint2 in human_skeleton.BodyParts:
                key2 = keypoint2.name

                for i in range(max_range):
                    if f"{key1}-{key2}" not in distances:
                        distances[f"{key1}-{key2}"] = []
                    distance = None
                    if key1 not in coordinates:
                        pass
                    elif None not in coordinates[key1][i] and None not in coordinates[key2][i]:
                        distance = np.linalg.norm(coordinates[key1][i] - coordinates[key2][i])
                    distances[f"{key1}-{key2}"].append(distance)

        for keypoint in coordinates:
            coordinates[keypoint] = np.array(coordinates[keypoint]).tolist()

        data = {
            'coordinates': coordinates,
            'velocities': velocities,
            'distances': distances,
            'timestamps': timestamps,
            'frames': frame_indexes
        }

        with open(f"tmp_data_{bag_filename.replace('/', '_')}.json", 'w+') as file:
            json.dump(data, file)
    else:
        data = feet_analysis.get_data("tmp_data" + bag_filename.replace('/', '_') + '.json')
        coordinates = data['coordinates']
        distances = data['distances']
        velocities = data['velocities']
        timestamps = data['timestamps']
        frame_indexes = data['frames']
        start_timestamp = timestamps[0]

    # print("Not saving to DB")
    # return
    walking_data = feet_analysis.preprocess_data_for_walk_analysis(data)
    number_of_frames = 0
    for i in range(len(walking_data['coordinates']['NOSE'])):
        number_of_frames += 1
        human = human_skeleton.HumanSkeleton()
        human.from_feet_analytics(walking_data['coordinates'], i)

        if median_center is not None:
            coord2D, coord3D = human.to_json(median_center, median_center + median_x,
                                             median_center + median_y,
                                             median_center + median_z,
                                             scale=0.21)
        else:
            coord2D, coord3D = human.to_json()  # If there is no Aruco box

        footsteps_per_velocity = 0
        stance_swing_ratio = None
        double_support_ratio = None
        walking_width = None
        walking_speed = 0.0
        footstep_length = 0.0

        if number_of_frames > 1:
            reduced_walking_data = feet_analysis.reduce_walking_data_size(walking_data, number_of_frames)
            footsteps_per_velocity = \
                feet_analysis.estimate_footsteps(reduced_walking_data, motion_threshold=motion_threshold)[0]
            try:
                stance_swing_ratio = feet_analysis.compute_stance_swing_ratio(reduced_walking_data,
                                                                            motion_threshold=motion_threshold)
            except:
                pass
            try:
                double_support_ratio = feet_analysis.compute_double_support_time(reduced_walking_data,
                                                                                 motion_threshold=motion_threshold)
            except:
                pass
            try:
                walking_width = feet_analysis.estimate_footsteps_width(reduced_walking_data)
            except:
                pass
            try:
                walking_speed = feet_analysis.compute_walking_speed(reduced_walking_data)
            except:
                pass
            try:
                distance_walked = (walking_speed * (timestamps[i] - start_timestamp)) / 1000
                if footsteps_per_velocity > 0:
                    footstep_length = distance_walked / footsteps_per_velocity
            except:
                pass

            client.insert_skeleton_keypoints(coord2D=coord2D, coord3D=coord3D, skeleton_id=i,
                                             timestamp=walking_data['timestamps'][i],
                                             walking_speed=walking_speed, footsteps_width=walking_width,
                                             number_of_footsteps=footsteps_per_velocity,
                                             stride_length=footstep_length * 2, step_length=footstep_length,
                                             device=device, send_immediatly=False)
    client.send_to_db()


    print("Done.")


def retrieve_data(bag_filename, client=None, verbose=False):
    if client is None:
        client = InfluxDB_connection.ListicClientDB(server_side=False)  # Connect locally if no InfluxDB client is given
        print("No specific InfluxDB client is given, continuing with local client")

    start = time.time()
    res2D, res3D = client.retrieve_skeleton_keypoints(bag_filename)
    end = time.time()
    if verbose:
        number_of_keypoints = 0
        start_time = math.inf
        end_time = -math.inf
        for skeleton in res2D:
            for key in skeleton:
                number_of_keypoints += len(skeleton[key])  # Number of frame per keypoint
                start_time = min(start_time, skeleton[key][0]['time_ms'])
                end_time = max(end_time, skeleton[key][-1]['time_ms'])
        # number_of_keypoints *= (3 + 2) # 3 coord for 3D + 2 coord for 2D
        print(
            f"Fetching {number_of_keypoints} keypoints : {round(end - start, 2)}s [duration = {(end_time - start_time)}s]")


if __name__ == "__main__":
    # Example of usage: python save_skeleton_to_server.py --dir recordings --local
    # or    python save_skeleton_to_server.py --dir recordings --remote --adress 192.168.143.29
    import argparse

    parser = argparse.ArgumentParser(
        description='Open all ros-bag files from a folder, run a skeleton detector on them and save the keypoints on the InfluxDB server')
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('-d', '--dir', dest='directory', metavar='directory', type=str,
                            help='Directory where the sequences are stored',
                            default="recordings")
    file_group.add_argument('-f', '--filename', dest='filename', metavar='filename', type=str, nargs="+",
                            help='File on which you want to run the skeleton detection and send keypoints')
    host_group = parser.add_mutually_exclusive_group(required=True)
    host_group.add_argument('--local', dest='local', help='Use the InfluxDB installed on this machine',
                            action='store_true')
    host_group.add_argument('--remote', dest='remote', help='Use the InfluxDB installed on a remote machine',
                            action='store_true')
    parser.add_argument('--adress', dest='adress', help='If using --remote, define the server adress', type=str)
    parser.add_argument('--port', dest='port', help='Define the port used by InfluxDB (default: 8086)', type=str,
                        default=8086)
    parser.add_argument('-l', '--login', dest='login', help='Username for InfluxDB', type=str)
    parser.add_argument('-p', '--password', dest='password', help='Password for InfluxDB', type=str)
    parser.add_argument('-db', '--database', dest='database', help='Database to connect to in InfluxDB', type=str)

    args = parser.parse_args()
    recordings_folder = args.directory
    local = args.local
    remote = args.remote
    adress = args.adress
    port = args.port
    login = args.login
    password = args.password
    database = args.database
    filename = args.filename

    if adress is None:
        if local:
            adress = 'localhost'
        elif remote:
            adress = '192.168.143.29' #parser.error('--adress is required if --remote is chosen')
    if login is not None and password is None:
        parser.error('--password is required with --login')

    db_client = InfluxDB_connection.ListicClientDB(server_side=remote, host=adress, port=port, login=login,
                                                   pwd=password, database=database)

    if filename is not None:
        files_list = filename
    else:
        files_list = glob.glob(recordings_folder + "/*.bag")

    print(f"{len(files_list)} files found")
    for file in files_list:
        print(" -", file)

    skeletrack = st.SkeletonTracker(choose_best_module=True, image_is_upside_down = True)
    for i, file in enumerate(files_list):
        print(f"{round((i / len(files_list)) * 100)}% ... {file}")
        save_data(file, client=db_client, skeletrack=skeletrack, use_aruco_coord = True)
    # retrieve_data(file, client = db_client)
    print("100%")
# for i, file in enumerate(glob.glob(recordings_folder + "/*.bag"), 1):
# 	plt.figure(i)
# 	retrieve_data(file, client = db_client)
# plt.show()
