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


class HumanStorage:
    def __init__(self, send_data_after=-1, db: InfluxDB_connection.ListicClientDB = None):
        self.humans = {}
        self.number_of_data_collected = 0
        self.send_data_after = send_data_after
        self.db = db

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
            for timestamp, human in self.humans[human_id].items():
                if starting_point is None:
                    starting_point = human.get_center_of_gravity()
                    start_timestamp = timestamp
                ending_point = human.get_center_of_gravity()
                end_timestamp = timestamp
                coord2D, coord3D = human.to_json(center, center + x, center + y, center + z,
                                                 scale=convert_realsense_to_real_world.marker_size)
                raw2D, raw3D = human.to_json()
                db.insert_skeleton_keypoints(coord2D, coord3D, timestamp, human_id, send_immediately=False,
                                             raw_coord_2D=raw2D, raw_coord_3D=raw3D)
            try:
                speed = np.linalg.norm(starting_point - ending_point)/((end_timestamp - start_timestamp)/1000)
                print(f"Speed of human #{human_id+1} = {speed} m/s")
            except:
                pass
        db.send_to_db()
        self.reset()

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

    # Start the Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    pipeline.start(config)

    # Create a skeleton tracker
    skeletrack = tracker.SkeletonTracker(choose_best_module=True, pipeline=pipeline, image_is_upside_down=True)

    # Instantiate the human storage to get the skeletons and send automatically to InfluxDB
    human_storage = HumanStorage(send_data_after=10, db=db_client)

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
            color_frame = convert_realsense_to_real_world.display_reference(color_frame, pipeline)

        else:
            calibration_finished = True
            skeletons = skeletrack.track_skeletons(color_frame)
            human_storage.update(skeletrack, depth_frame)

            if skeletrack.check_user_presence():
                body_present = True
                color_frame = skeletrack.draw_skeleton(color_frame)
            else:
                body_present = False

        status = ""
        status_text = ""
        status_color = (0, 0, 0)
        if not calibration_finished:
            status = "calib"
            status_text = f"Calibrating camera... ({round(calibrations_remaining/20)}s)"
            status_color = (255, 0, 0) # RED
        elif body_present:
            status = "record"
            status_text = "Recording skeleton"
            status_color = (0, 255, 0) # GREEN
        else:
            status = "wait"
            status_text = "No skeleton found"
            status_color = (0, 0, 255) # BLUE
        # Display the FPS and status on screen
        cv2.putText(color_frame, "Status: " + status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(color_frame, "FPS: " + str(round(fps, 2)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        color_frame = cv2.resize(color_frame, (1200, 600))
        cv2.imshow('Realsense', cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

        # Exit with 'Q' or 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    pipeline.stop()
    print(" --- End ---")
