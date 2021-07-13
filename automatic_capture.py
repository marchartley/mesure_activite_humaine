"""! The automatic capture system that start a recording at the moment somebody enters the field of view
of the camera
"""

# Imports
import skeleton_tracker
import skeleton_tracker as tracker
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
import recording
import threading


class AutomaticCapture:
    """! The Automatic capture will use the output from the skeleton tracker to start or not the recording of a sequence

    It will check if a person is present on screen using threads to avoid slowing down the program.
    The 'latency' parameter is the elapsed time in seconds between two checks
    """

    def __init__(self, enable: bool = True, tracker: skeleton_tracker.SkeletonTracker = None,
                 pipeline: rs.pipeline = None,
                 recording_dir: str = "", latency: float = 3.0):
        """! The Automatic Capture constructor
        @param enable Starts directly the automatic capture system or not
        @param tracker The skeleton tracker used to find humans in the camera field of view
        @param pipeline The RealSense pipeline that allows to manipulate the camera data
        @param recording_dir The folder in which you want to save the rosbag files
        @param latency The offset time during which the system continues to record even if nobody is on screen

        @return An instance of the Automatic Capture
        """
        ## Boolean that define if the automatic capture is on or not.
        self.AUTOMATIC_CAPTURE = enable
        ## The tracker is the skeleton tracker that find skeletons in an image
        self.tracker = tracker
        self.associate_tracker(tracker)
        ## Pipeline defined from realsense that transform data from the camera into an image using their SDK
        self.pipeline = pipeline
        self.associate_pipeline(pipeline)

        ## Variable that store whever a person is currently on screen
        # This variable should not be manipulated from outside the class
        self._body_detected = False
        ## The time the system continues to capture if nobody is on screen. This is useful if a person could reappear
        # few seconds after being in a blind spot
        self.latency = latency
        ## Time at which the last person was seen on screen.
        # This time will be used to stop the capture at the time "_last_seen_time + latency"
        self._last_seen_time = time.time()
        ## Is the system currently recording or not
        self._currently_capturing = False
        ## Folder in which we save the recordings
        self.recording_dir = recording_dir

        # self.detection_thread = None

    def associate_tracker(self, tracker: skeleton_tracker.SkeletonTracker):
        """! Function to associate a new tracker to the automatic capture
        to be able to find a person in the screen.

        @param tracker The skeleton tracker that will be used

        @note This will add a callback to the tracker so the automatic capture will receive events when
        somebody is on screen
        """
        self.tracker = tracker
        if tracker is not None:
            # If we give a real tracker, add a callback when an image is being processed to execute the body check
            self.tracker.add_callback('process', self._check_if_body_found)

    def associate_pipeline(self, pipeline):
        """! Function to associate a new RealSense pipeline to the Automatic Capture system.
        This can be used in order to change the dedicated camera, for instance.
        """
        self.pipeline = pipeline

    def set_automatic_capture(self, isAutomaticCapture):
        """! Function to enable or disable to automatic capture system.
        @see enable_automatic_capture() and disable_automatic_capture()"""
        self.AUTOMATIC_CAPTURE = isAutomaticCapture

    def enable_automatic_capture(self):
        """! Function to enable the automatic capture system"""
        self.set_automatic_capture(True)

    def disable_automatic_capture(self):
        """! Function to disable the automatic capture system"""
        self.set_automatic_capture(False)

    def _check_if_body_found(self, skeleton_results):
        """! Function that will be called at each frame that the skeleton tracker will process
        It will modify the private attribute "_body_detected" depending on the result.
        """
        # Check for a body from the results of the tracker
        # At least 3 keypoints found with > 0.5 confidence
        self._body_detected = self.tracker.check_user_presence(tracker_results=skeleton_results,
                                                               required_confidence=0.5,
                                                               min_number_of_joints_per_person=3)

    # This function has to bee called in the main loop to start/stop the recording
    def update(self):
        """! Function to call at each frame to let the automatic capture system check
        whever it has to continue/start the recording

        @return A boolean that indicates whever a person is in sight or not.
        """
        # If the system is enabled and somebody's in sight, continue/start the recording
        # Otherwise, check if we have to stop it if the "latency" is already finished
        if self.AUTOMATIC_CAPTURE:
            if self._body_detected:
                if not self.isCapturing():
                    self.capture()
                self._last_seen_time = time.time()
            else:
                current_time = time.time()
                if current_time - self._last_seen_time >= self.latency:
                    if self.isCapturing():
                        self.stop_capture()
        return self._body_detected

    def capture(self):
        """! Function to start the recording if it hasn't began already
        @note A pipeline must be set in order to record anything
        """
        if self.pipeline is not None:
            if not self._currently_capturing:
                print("Start capturing")
                current_time = time.time()
                filename = self.recording_dir + "/"
                filename += time.strftime("%Y-%m-%d_%H_%M_%S_")
                filename += str(int((current_time - int(current_time)) * 1000)) + ".bag"
                # Filename is defined from the timestamp, then start the recording process
                self.pipeline = recording.start_recording(filename, self.pipeline)
                self._currently_capturing = True

    def stop_capture(self):
        """! Function to stop the recording if it has been started
        @note A pipeline must be set to start/stop a recording
        """
        if self.pipeline is not None:
            if self._currently_capturing:
                # If there is a pipeline and it is currently capturing, stop the process
                self.pipeline = recording.stop_recording(self.pipeline)
                self._currently_capturing = False

    def isCapturing(self):
        """! Check if the system is currently capturing
        """
        return self._currently_capturing


def main():
    # Example of usage: python automatic_capture.py --dir recordings --time-margin 5
    import argparse

    parser = argparse.ArgumentParser(
        description='Enable the RealSense camera and record a sequence if a person is detected')
    parser.add_argument('-d', '--dir', dest='directory', metavar='directory', type=str,
                        help='Directory where to save the sequences (current directory by default)',
                        default=".", required=False)
    parser.add_argument('-m', '--time-margin', dest='margin',
                        help='Time (in second) elapsed without seeing anyone before stopping the recording',
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

    # Enable the auto-capture
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


if __name__ == "__main__":
    main()
