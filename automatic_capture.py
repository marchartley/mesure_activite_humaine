import skeleton_tracker as tracker
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
import recording
import threading


# The Automatic capture will use the output from the skeleton tracker to start or not the recording of a sequence
# The 'latency' parameter is the elapsed time in seconds between two checks
class AutomaticCapture:
    def __init__(self, enable=True, tracker=None, pipeline=None, recording_dir="", latency=3):  # Latency in sec
        self.AUTOMATIC_CAPTURE = enable
        self.tracker = None
        self.associate_tracker(tracker)
        self.pipeline = None
        self.associate_pipeline(pipeline)

        self._body_detected = False
        self.latency = latency
        self.last_seen_time = time.time()
        self._currently_capturing = False
        self.recording_dir = recording_dir

        self.detection_thread = None

    def associate_tracker(self, tracker):
        self.tracker = tracker
        if tracker is not None:
            self.tracker.add_callback('process', self._check_if_body_found)

    def associate_pipeline(self, pipeline):
        self.pipeline = pipeline

    def set_automatic_capture(self, isAutoaticCapture):
        self.AUTOMATIC_CAPTURE = isAutoaticCapture

    def enable_automatic_capture(self):
        self.set_automatic_capture(True)

    def disable_automatic_capture(self):
        self.set_automatic_capture(False)

    def _check_if_body_found(self, skeleton_results):
        # The process should be done with threads to keep a good frame rate, but I had problem with it
        # 	if self.detection_thread is None or not self.detection_thread.is_alive():
        # 		self.detection_thread = threading.Thread(target = self._check_if_body_found_async, kwargs = {'skeleton_results': skeleton_results}, daemon = False)
        # 		self.detection_thread.start()
        #
        # Check for a body in an synchronous way
        # At least 3 keypoints found with > 0.5 confidence
        self._body_detected = self.tracker.check_user_presence(tracker_results=skeleton_results,
                                                               required_confidence=0.5,
                                                               min_number_of_joints_per_person=3)

    # This function has to bee called in the main loop to start/stop the recording
    def update(self):
        if self.AUTOMATIC_CAPTURE:
            if self._body_detected:
                if not self.isCapturing():
                    self.capture()
                self.last_seen_time = time.time()
            else:
                current_time = time.time()
                if current_time - self.last_seen_time >= self.latency:
                    if self.isCapturing():
                        self.stop_capture()
        return self._body_detected

    # Start recording (if not already)
    def capture(self):
        if self.pipeline is not None:
            if not self._currently_capturing:
                print("Start capturing")
                current_time = time.time()
                filename = self.recording_dir + "/"
                filename += time.strftime("%Y-%m-%d_%H_%M_%S_")
                filename += str(int((current_time - int(current_time)) * 1000)) + ".bag"
                self.pipeline = recording.start_recording(filename, self.pipeline)
                self._currently_capturing = True

    # Stop the recording if not already done
    def stop_capture(self):
        if self.pipeline is not None:
            if self._currently_capturing:
                self.pipeline = recording.stop_recording(self.pipeline)
                self._currently_capturing = False

    # Check if the device is recording
    def isCapturing(self):
        return self._currently_capturing


if __name__ == "__main__":
    # Example of usage: python automatic_capture.py --dir recordings --time-margin 5
    import argparse

    parser = argparse.ArgumentParser(
        description='Enable the RealSense camera and record a sequence if a person is detected')
    parser.add_argument('-d', '--dir', dest='directory', metavar='directory', type=str,
                        help='Directory where to save the sequences (current directory by default)',
                        default=".", required=False)
    parser.add_argument('-m', '--time-margin', dest='margin',
                        help='Time (in second) elpased without seeing anyone before stopping the recording',
                        type=float, default=10.0)

    args = parser.parse_args()
    directory = args.directory
    time_epsilon = args.margin

    # Start the Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    pipeline.start(config)

    # Create a skeleton tracker
    skeletrack = tracker.SkeletonTracker(choose_best_module=True)

    # Enable the auto-capturer
    auto_cap = AutomaticCapture(tracker=skeletrack, pipeline=pipeline, latency=time_epsilon, recording_dir=directory,
                                enable=True)
    number_of_frames = 0
    start_time = time.time()
    fps = 0

    # Due to some latency found on my previous computer, I had to do the human checking on a even lower frequency
    next_check = time.time()

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
        depth_frame = np.asanyarray(frames.get_depth_frame().get_data()).copy()

        # Check for a human asynchronously (with threads) to avoid loosing FPS
        skeletons = skeletrack.track_skeletons_async(color_frame, min_time_start_process=next_check)
        if time.time() > next_check:
            next_check += 5.0
        # Start/stop the recording depending on the result
        auto_cap.update()

        # Display the FPS and status on screen
        cv2.putText(color_frame, "Status: " + ("recording" if auto_cap.isCapturing() else "waiting..."), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if auto_cap.isCapturing() else (0, 255, 0)), 2)
        cv2.putText(color_frame, "FPS: " + str(round(fps, 2)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if auto_cap.isCapturing() else (0, 255, 0)), 2)
        cv2.imshow('Realsense', cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

        # Exit with 'Q' or 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    pipeline.stop()
    print(" --- End ---")
