"""! This script can be used to capture and save the skeletons of person walking in front of a camera"""

import datetime

import dateutil.parser

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

## The last value estimated for the person's speed to display it on screen
last_speed_registered = 0
## The last time we've computed the person's speed to display it on screen
last_date_speed_registered = None

class HumanStorage:
    """! A buffer class that store the coordinates of the persons on screen for a moment and send all of it
    at once into an InfluxDB database
    """
    def __init__(self, send_data_after: int = -1, db: InfluxDB_connection.ListicClientDB = None, cooldown: int = None,
                 sensors_height: str = 'hips'):
        """! Constructor
        @param send_data_after Defines the number of skeletons the class can contain before flushing everything into
        the InfluxDB database. Keep it to -1 to flush only when the cooldown is finished (much better method)
        @param db The client used to access the database
        @param cooldown The number of frames without anybody on screen after which the data is sent to the database
        @param sensors_height Position of the sensors when filming the scene

        @return An instance of the HumanStorage
        """
        ## A collection of skeletons
        self.humans = {}
        ## Keeping track of the number of data collected to compute the cooldown
        self.number_of_data_collected = 0
        ## The number of skeletons this object should keep in memory before sending everything into the DB
        self.send_data_after = send_data_after
        ## The client that access the database
        self.db = db
        ## Position of the sensors, useful when saving the data
        self.sensors_height = sensors_height

        ## The number of frames required without anybody in sight before sending to DB
        self.cooldown_limit = cooldown
        ## The current number of frames left before sending to DB
        self.cooldown = self.cooldown_limit

    def update(self, skeletrack: tracker.SkeletonTracker, depth_map: rs.depth_frame,
               db: InfluxDB_connection.ListicClientDB = None) -> None:
        """! Take the skeletons found by a skeleton tracker and store them in an array. Send data to DB if needed.
        @param skeletrack A skeletrack object that analysed a frame
        @param depth_map The depth map captured by the camera
        @param db If no database client have been given until now, you can pass the client.
        """
        number_of_persons = skeletrack.get_number_of_skeletons()
        timestamp = recording.get_frame_timestamp(depth_map)

        if number_of_persons > 0:
            self.number_of_data_collected += 1

        for id_skeleton in range(number_of_persons):
            # Create a HumanSkeleton object that will use the skeleton tracker to get initialized
            human = human_skeleton.HumanSkeleton()
            human.from_tracker(skeletrack, id_skeleton, depth_map)
            # Add the HumanSkeleton in the array
            if id_skeleton not in self.humans:
                self.humans[id_skeleton] = {}
            self.humans[id_skeleton][timestamp] = human

        # If we need to send the data after some time, check if this time is done
        if self.send_data_after > 0:
            if db is None:
                db = self.db
            if self.number_of_data_collected >= self.send_data_after:
                self.toDB(db)
        # Also, check if the cooldown is done
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

    def setDB(self, db: InfluxDB_connection.ListicClientDB) -> None:
        """! Update the Database Client. Nothing more"""
        self.db = db

    def toDB(self, db: InfluxDB_connection.ListicClientDB = None) -> None:
        """! Send all the data stored into the database thanks to the client
        @param db If you haven't provided a client yet, you can pass one now
        """
        # Take the position of the referential (Aruco box)
        center, x, y, z = convert_realsense_to_real_world.get_aruco_reference()
        if db is None:
            db = self.db
        if db is None:  # If the parameter was None and there was no DB given from constructor
            raise Exception("Skeletons were asked to be send to database but no InfluxDB client has been provided")

        # Send data for each person.
        for human_id in self.humans:

            for i, (timestamp, human) in enumerate(self.humans[human_id].items()):
                coord2D, coord3D = human.to_json(center, center + x, center + y, center + z,
                                                 scale=convert_realsense_to_real_world.marker_size)
                raw2D, raw3D = human.to_json()
                # Send to the database the translated coordinates AND the original coordinates, just in case...
                db.insert_skeleton_keypoints(coord2D, coord3D, timestamp, human_id, send_immediately=False,
                                             raw_coord_2D=raw2D, raw_coord_3D=raw3D)

            try:
                # If you wish you can display all the walking features of the person.
                # But it wasn't really visible on screen, so I kinda stopped that
                speed = feet_analysis.compute_walking_speed(self.humans[human_id])
                print(f"Speed of human #{human_id+1} = {speed} m/s")
                data = feet_analysis.create_feet_analysis_data_from_humans(self.humans[human_id])
                motion_threshold = 0.01  # 0.01 m/s ~= 0.04km/h

                original_data = feet_analysis.preprocess_data_for_walk_analysis(data)

                footsteps_motion = feet_analysis.estimate_footsteps(original_data, motion_threshold=motion_threshold)[0]
                footsteps_distance = feet_analysis.estimate_footsteps_by_distance(original_data, 3)[0]
                cycle_time = feet_analysis.compute_cycle_time(data)
                stance_ratio_right = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                                                                        concerned_foot="RIGHT_ANKLE")
                stance_ratio_left = feet_analysis.compute_stance_ratio(data, motion_threshold=motion_threshold,
                                                                       concerned_foot="LEFT_ANKLE")
                double_support_ratio = feet_analysis.compute_double_support_time(data,
                                                                                 motion_threshold=motion_threshold)
                footstep_width = feet_analysis.estimate_footsteps_width(data)
                walking_speed = feet_analysis.compute_walking_speed(data)

                stride_length, right_length, left_length = feet_analysis.compute_stride_length(data)

                walking_direction = feet_analysis.get_walking_direction(data)

                print(footsteps_motion, "footsteps estimated with motion")
                print(footsteps_distance, "footsteps estimated with distance")

                print(f"Walking speed              = {round(walking_speed * 3.6, 2)} km/h")
                print(f"Cycle Time                 = {round(cycle_time, 3)} sec/step")
                print(f"Cadence                    = {round(60 / cycle_time, 1)} steps/min")
                print(f"Stance ratio (Right)       = {round(stance_ratio_right * 100, 2)}%")
                print(f"Stance ratio (Left)        = {round(stance_ratio_left * 100, 2)}%")
                print(f"Double support time        = {round(double_support_ratio * 100, 2)}%")
                print(f"Width of footsteps         = {round(footstep_width * 100)}cm")
                print(f"Stride length              = {round(stride_length * 100)}cm")
                print(f"Stride length (R)          = {round(right_length * 100)}cm")
                print(f"Stride length (L)          = {round(left_length * 100)}cm")
                if walking_direction == 1:
                    print(f"Walking from North to South")
                elif walking_direction == -1:
                    print(f"Walking from South to North")
                else:
                    print(f"Not walking through the corridor")

                feet_analysis.visualize_data(original_data, time_in_seconds=True, non_blocking = True)
            except Exception as e:
                print(e)
                pass
        # Flush the db client
        db.send_to_db()
        # Empty the storage
        self.reset()
        # Update the datasets on a thread to avoid blocking the application
        threading.Thread(target=update_datasets, kwargs={'use_cache': False, 'db_client': self.db}).start()

    def reset(self):
        """! Remove all data stored"""
        self.humans = {}
        self.number_of_data_collected = 0


def update_datasets(db_client: InfluxDB_connection.ListicClientDB, use_cache: bool = False) -> None:
    """! Update the datasets used for the deep learning training
    @param db_client The client that can access the database
    @param use_cache Defines if the client can use the cache. It's not a good idea in this case I think...
    """
    # Update/Create the CSV with all the PIR values
    pir_data = create_pir_csv.fill_csv(create_CSV_from_scratch=False,
                                       output_filename='gait_values_ankles.csv',
                                       # start_time=datetime.datetime(year=2021, month=6, day=4).timestamp() * 1000,
                                       start_time=datetime.datetime(year=2021, month=6, day=16).timestamp() * 1000,
                                       client=db_client,
                                       use_cache=use_cache,
                                       sensor_height='ankles')
    if pir_data is not None:
        # Update some variables, just for display
        global last_speed_registered, last_date_speed_registered
        last_speed_registered = pir_data['walking_speed'].values[-1]
        last_date_speed_registered = dateutil.parser.parse(pir_data['time'].values[-1]).timestamp()
        # Update the CWT dataset
        deep_neural_training.precompute_morlet_transforms(pir_data,
                                                          'all_pirs_transformed_ankles.data', output_filename=True)


def main():
    # Example of usage: python direct_camera_to_db.py --local --sensor ankles
    #  or
    #                   python direct_camera_to_db.py --remote --sensor hips
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
    parser.add_argument('--test-model', dest='test_model', help='Use a pretrained deep learning model to check on '
                                                                'live data', action='store_true')
    parser.add_argument('--sensor', dest='sensors_height',
                        help='Position of the sensors (hips, ankles, one_sided, ...)')

    args = parser.parse_args()
    local = args.local
    remote = args.remote
    address = args.address
    port = args.port
    login = args.login
    password = args.password
    database = args.database
    calibration_time = args.calibration_time
    sensor = args.sensors_height

    number_of_aruco_calibration_needed = calibration_time * 20  # 20 is an arbitrary value, estimating that
    #                                                               we catch the aruco cube 20 times per second

    if address is None:
        if local:
            address = 'localhost'
        elif remote:
            address = '192.168.143.29'  # parser.error('--address is required if --remote is chosen')
    if login is not None and password is None:
        parser.error('--password is required with --login')

    db_client = InfluxDB_connection.ListicClientDB(server_side=remote, host=address, port=port, login=login,
                                                   pwd=password, database=database)

    threading.Thread(target=update_datasets, kwargs={'use_cache': True, 'db_client': db_client}).start()

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
        if last_date_speed_registered is not None:
            time_diff = time.time() - last_date_speed_registered
            time_h = int((time_diff // 60) // 60)
            time_m = int(time_diff // 60)
            time_s = int(time_diff % 60)
            str_time = f"{time_h}h{time_m}m{time_s}s" if time_h > 0 else f"{time_m}m{time_s}s" if time_m > 0 else f"{time_s}s"
            speed_text = f"Last speed: {round(last_speed_registered, 2)}m/s ({str_time} ago)"
            cv2.putText(color_frame, speed_text, (10, 70),
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


if __name__ == "__main__":
    main()
