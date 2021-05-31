import feet_analysis
import human_skeleton
import skeleton_tracker as tracker
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
import recording
import threading
import InfluxDB_connection
import convert_realsense_to_real_world
# import matplotlib.pyplot as plt
import threading
import create_pir_csv
import deep_neural_training
import utils


class HumanStorage:
    def __init__(self, send_data_after=-1, db: InfluxDB_connection.ListicClientDB = None, cooldown=None):
        self.humans = {}
        self.number_of_data_collected = 0
        self.send_data_after = send_data_after
        self.db = db

        self.cooldown_limit = cooldown
        self.cooldown = self.cooldown_limit

    def update(self, skeletrack: tracker.SkeletonTracker, depth_map, db: InfluxDB_connection.ListicClientDB = None):
        self.number_of_data_collected += 1
        number_of_persons = skeletrack.get_number_of_skeletons()
        timestamp = recording.get_frame_timestamp(depth_map)

        for id_skeleton in range(number_of_persons):
            human = human_skeleton.HumanSkeleton()
            human.from_tracker(skeletrack, id_skeleton, depth_map)
            if id_skeleton not in self.humans:
                self.humans[id_skeleton] = {}
            self.humans[id_skeleton][timestamp] = human

        if self.send_data_after > 0:
            if db is None:
                db = self.db
            if self.number_of_data_collected >= self.send_data_after:
                self.toDB(db)
        elif self.cooldown is not None:
            if number_of_persons == 0 and len(self.humans) > 0:
                self.cooldown -= 1
            else:
                self.cooldown = self.cooldown_limit

            if db is None:
                db = self.db
            if self.cooldown <= 0:
                self.toDB(db)
                self.cooldown = self.cooldown_limit

    def setDB(self, db: InfluxDB_connection.ListicClientDB):
        self.db = db

    def toDB(self, db: InfluxDB_connection.ListicClientDB = None):
        center, x, y, z = convert_realsense_to_real_world.get_aruco_reference()
        if db is None:
            db = self.db
        if db is None:  # If the parameter was None and there was no DB given from constructor
            raise Exception("Skeletons were asked to be send to database but no InfluxDB client has been provided")

        for human_id in self.humans:
            starting_point = None
            ending_point = None
            start_timestamp = None
            end_timestamp = None

            for i, (timestamp, human) in enumerate(self.humans[human_id].items()):
                if starting_point is None:
                    starting_point = human.get_center_of_gravity()
                    start_timestamp = timestamp
                ending_point = human.get_center_of_gravity()
                end_timestamp = timestamp
                # coord2D, coord3D = human.apply_new_reference(convert_realsense_to_real_world.get_aruco_reference(),
                #                                              convert_realsense_to_real_world.marker_size).to_json()
                coord2D, coord3D = human.to_json(center, center + x, center + y, center + z,
                                                 scale=convert_realsense_to_real_world.marker_size)
                raw2D, raw3D = human.to_json()
                db.insert_skeleton_keypoints(coord2D, coord3D, timestamp, human_id, send_immediately=False,
                                             raw_coord_2D=raw2D, raw_coord_3D=raw3D)

            try:
                pass
                # speed = np.linalg.norm(starting_point - ending_point)/((end_timestamp - start_timestamp)/1000)
                # print(f"Speed of human #{human_id+1} = {speed} m/s")
                # data = feet_analysis.create_feet_analysis_data_from_humans(self.humans[human_id])
                # motion_threshold = 0.8  # 0.8 m/s ~= 3km/h
                #
                # original_data = feet_analysis.preprocess_data_for_walk_analysis(data)
                #
                # footsteps_motion = feet_analysis.estimate_footsteps(original_data, motion_threshold=motion_threshold)[0]
                # footsteps_distance = feet_analysis.estimate_footsteps_by_distance(original_data, 3)[0]
                # cycle_time = feet_analysis.compute_cycle_time(data)
                # stance_ratio_right = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                #                                                         concerned_foot="RIGHT_ANKLE")
                # stance_ratio_left = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                #                                                        concerned_foot="LEFT_ANKLE")
                # double_support_ratio = feet_analysis.compute_double_support_time(data,
                #                                                                  motion_threshold=motion_threshold)
                # footstep_width = feet_analysis.estimate_footsteps_width(data)
                # walking_speed = feet_analysis.compute_walking_speed(data)
                # stride_length, right_length, left_length = feet_analysis.compute_stride_length(data)
                #
                # walking_direction = feet_analysis.get_walking_direction(data)
                #
                # print(footsteps_motion, "footsteps estimated with motion")
                # print(footsteps_distance, "footsteps estimated with distance")
                #
                # print(f"Walking speed              = {round(walking_speed * 3.6, 2)} km/h")
                # print(f"Cycle Time                 = {round(cycle_time, 3)} sec/step")
                # print(f"Cadence                    = {round(60 / cycle_time, 1)} steps/min")
                # print(f"Stance ratio (Right)       = {round(stance_ratio_right * 100, 2)}%")
                # print(f"Stance ratio (Left)        = {round(stance_ratio_left * 100, 2)}%")
                # print(f"Double support time        = {round(double_support_ratio * 100, 2)}%")
                # print(f"Width of footsteps         = {round(footstep_width * 100)}cm")
                # print(f"Stride length              = {round(stride_length * 100)}cm")
                # print(f"Stride length (R)          = {round(right_length * 100)}cm")
                # print(f"Stride length (L)          = {round(left_length * 100)}cm")
                # if walking_direction == 1:
                #     print(f"Walking from North to South")
                # elif walking_direction == -1:
                #     print(f"Walking from South to North")
                # else:
                #     print(f"Not walking through the corridor")

                # feet_analysis.visualize_data(original_data, time_in_seconds=True, non_blocking = True)
            except Exception as e:
                print(e)
                pass
        db.send_to_db()
        self.reset()
        pir_data = create_pir_csv.fill_csv(create_CSV_from_scratch=False,
                                           output_filename='gait_values.csv',
                                           client=db,
                                           verbose=True,
                                           use_cache=False)
        deep_neural_training.precompute_morlet_transforms(pir_data, 'all_pirs_transformed.data')

    def reset(self):
        self.humans = {}
        self.number_of_data_collected = 0


if __name__ == "__main__":

    # Example of usage: python direct_camera_to_db.py --local
    #  or
    #                   python direct_camera_to_db.py --remote
    import argparse

    parser = argparse.ArgumentParser(description='Save the skeletons detected on the camera without saving the rosbag '
                                                 'file')
    host_group = parser.add_mutually_exclusive_group(required=True)
    host_group.add_argument('--local', dest='local', help='Use the InfluxDB installed on this machine',
                            action='store_true')
    host_group.add_argument('--remote', dest='remote', help='Use the InfluxDB installed on a remote machine',
                            action='store_true')
    parser.add_argument('--address', dest='address', help='If using --remote, define the server address', type=str)
    parser.add_argument('--port', dest='port', help='Define the port used by InfluxDB (default: 8086)', type=str,
                        default=8086)
    parser.add_argument('-l', '--login', dest='login', help='Username for InfluxDB', type=str)
    parser.add_argument('-p', '--password', dest='password', help='Password for InfluxDB', type=str)
    parser.add_argument('-db', '--database', dest='database', help='Database to connect to in InfluxDB', type=str)
    parser.add_argument('--calibration-time', dest='calibration_time', help='Time used to calibrate the Aruco box. '
                                                                            'This is an estimation as it will be '
                                                                            'internally converted as a number of '
                                                                            'frames. 10 seconds is a recommendation.',
                        default=10.0, type=float)

    args = parser.parse_args()
    local = args.local
    remote = args.remote
    address = args.address
    port = args.port
    login = args.login
    password = args.password
    database = args.database
    calibration_time = args.calibration_time

    number_of_aruco_calibration_needed = calibration_time * 20  # 10 is an arbitrary value, estimating that
    #                                                               we catch the aruco cube 10 times per second

    if address is None:
        if local:
            address = 'localhost'
        elif remote:
            address = '192.168.143.29'  # parser.error('--address is required if --remote is chosen')
    if login is not None and password is None:
        parser.error('--password is required with --login')

    db_client = InfluxDB_connection.ListicClientDB(server_side=remote, host=address, port=port, login=login,
                                                   pwd=password, database=database)

    def update_datasets():
        pir_data = create_pir_csv.fill_csv(create_CSV_from_scratch=False,
                                output_filename='gait_values.csv',
                                client=db_client,
                                verbose=True)
        deep_neural_training.precompute_morlet_transforms(pir_data, 'all_pirs_transformed.data')

    threading.Thread(target=update_datasets).start()

    # Start the Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    pipeline.start(config)

    # Create a skeleton tracker
    skeletrack = tracker.SkeletonTracker(choose_best_module=True, pipeline=pipeline)  # , image_is_upside_down=True)

    # Instantiate the human storage to get the skeletons and send automatically to InfluxDB
    # Cooldown is the number of frames without anybody on screen before sending to InfluxDB
    human_storage = HumanStorage(cooldown=100, db=db_client)

    number_of_frames = 0
    start_time = time.time()
    fps = 0

    body_present = False
    calibration_finished = False

    while True:
        try:
            frames = pipeline.wait_for_frames()
        except:
            break
        number_of_frames += 1
        # Compute current FPS
        if (time.time() - start_time) > 1:
            fps = number_of_frames / (time.time() - start_time)
            number_of_frames = 0
            start_time = time.time()

        color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth_frame = frames.get_depth_frame()

        calibrations_remaining = number_of_aruco_calibration_needed - convert_realsense_to_real_world.get_number_of_calibration_of_aruco_box()
        if convert_realsense_to_real_world.get_number_of_calibration_of_aruco_box() < number_of_aruco_calibration_needed:
            convert_realsense_to_real_world.calibrate_aruco_cube(color_frame, depth_frame=depth_frame,
                                                                 pipeline=pipeline)

        else:
            calibration_finished = True
            skeletons = skeletrack.track_skeletons(color_frame)
            human_storage.update(skeletrack, depth_frame)

            if skeletrack.check_user_presence():
                body_present = True
                unavailable_indices = []
                for i, id_skeleton in enumerate(skeletons['ids']):
                    human = human_skeleton.HumanSkeleton()
                    human.from_tracker(skeletrack, i, depth_frame)
                    if not np.any([human.check_keypoints3D(key.name) for key in human_skeleton.BodyParts]):
                        unavailable_indices.append(i)
                color_frame = skeletrack.draw_skeleton(color_frame, index_of_unavailable=unavailable_indices)
                if len(unavailable_indices) == len(skeletons['ids']):
                    body_present = False
            else:
                body_present = False
        color_frame = convert_realsense_to_real_world.display_reference(color_frame, pipeline)

        status = ""
        status_text = ""
        status_color = (0, 0, 0)
        if not calibration_finished:
            status = "calib"
            status_text = f"Calibrating camera... ({round(calibrations_remaining / 20)}s)"
            status_color = (255, 0, 0)  # RED
        elif body_present:
            status = "record"
            status_text = "Recording skeleton"
            status_color = (0, 255, 0)  # GREEN
        else:
            status = "wait"
            status_text = "No skeleton seen"
            if human_storage.cooldown < human_storage.cooldown_limit:
                status_text += f' (saving data in {round(human_storage.cooldown / fps, 1)}s)'
            status_color = (0, 0, 255)  # BLUE

        # Rotate the image
        # color_frame = cv2.flip(color_frame, 0)
        # color_frame = cv2.flip(color_frame, 1)
        # Display the FPS and status on screen
        cv2.putText(color_frame, "Status: " + status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(color_frame, "FPS: " + str(round(fps, 2)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # color_frame = np.array(rs.colorizer().colorize(depth_frame).get_data())
        color_frame = cv2.resize(color_frame, (1200, 600))
        cv2.imshow('Realsense', cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

        # Exit with 'Q' or 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    pipeline.stop()
    print(" --- End ---")
