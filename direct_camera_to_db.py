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
        if db is None:
            db = self.db
        if db is None: # If the parameter was None and there was no DB given from constructor
            raise Exception("Skeletons were asked to be send to database but no InfluxDB client has been provided")
        for human_id in self.humans:
            for timestamp, human in self.humans[human_id].items():
                coord2D, coord3D = human.to_json()
                db.insert_skeleton_keypoints(coord2D, coord3D, timestamp, human_id, send_immediatly=False)
        db.send_to_db()
        self.reset()
        print("sent")

    def reset(self):
        self.humans = {}
        self.number_of_data_collected = 0

if __name__ == "__main__":
    client = InfluxDB_connection.ListicClientDB(server_side=False)
    # Start the Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    pipeline.start(config)

    # Create a skeleton tracker
    skeletrack = tracker.SkeletonTracker(choose_best_module=True, pipeline=pipeline)
    human_storage = HumanStorage(send_data_after=30, db = client)

    number_of_frames = 0
    start_time = time.time()
    fps = 0

    body_present = False

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

        skeletons = skeletrack.track_skeletons(color_frame)
        human_storage.update(skeletrack, depth_frame)

        if skeletrack.check_user_presence():
            body_present = True
        else:
            body_present = False

        # Display the FPS and status on screen
        cv2.putText(color_frame, "Status: " + ("recording" if body_present else "waiting..."), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if body_present else (0, 255, 0)), 2)
        cv2.putText(color_frame, "FPS: " + str(round(fps, 2)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        color_frame = skeletrack.draw_skeleton(color_frame)
        cv2.imshow('Realsense', cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

        # Exit with 'Q' or 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    pipeline.stop()
    print(" --- End ---")
