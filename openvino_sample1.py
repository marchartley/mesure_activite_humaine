import time

import pyrealsense2 as rs
import person_identification as identification
import cv2
import face_recognition
import recording

import numpy as np


if __name__ == "__main__":

	align_to = rs.stream.depth
	align = rs.align(align_to)

	pipeline = rs.pipeline()
	config = rs.config()
	recording.configure_streams(config)
	pipeline.start(config) # Live mode

	# backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history=500)
	num_frame = 0

	start_time = time.time()
	human = None
	playing = True

	current_timestamp = 0
	previous_timestamp = 0
	# automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline)
	found_person_name = ""
	proba = 0.0

	face_locations = []
	face_distances = np.array([])
	number_of_faces_detection = 0

	min_height = 1000

	mini = np.zeros((1, 1, 3))

	while True:
		if playing:
			previous_timestamp = current_timestamp
			try:
				frames = pipeline.wait_for_frames()
			except Exception as e:
				print(e)
				break
			num_frame += 1
			# frames = align.process(frames)
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			current_timestamp = recording.get_frame_timestamp(frames)

			color_frame = np.asanyarray(color_frame.get_data()).copy()
			color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
			rgb_small_frame = cv2.resize(color_frame.copy(), (0, 0), fx = 1 / 4, fy = 1 / 4)

			face_locations = face_recognition.face_locations(color_frame)
			# face_encodings = face_recognition.face_encodings(rgb_small_frame, model = 'small')

			color_frame = cv2.putText(color_frame, str(round((1 - proba) * 100, 2)) + "%: " + found_person_name,
			                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		#
			top, right, bottom, left = 0, 1, 1, 0
			if len(face_locations) > 0:
				if face_locations[0] is not None:
					top, right, bottom, left = face_locations[0]
					min_height = min(min_height, abs(bottom - top))
					mini = color_frame[top:bottom, left:right]
			mini = cv2.resize(mini, (800, 600))
			cv2.imshow("Face", mini)
			cv2.imshow("Full", color_frame)
		#
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break
		# elif k == ord('p'):
		# 	if playing:
		# 		pipeline = recording.pause_playback(pipeline)
		# 	else:
		# 		pipeline = recording.resume_playback(pipeline)
		# 	playing = not playing

	cv2.destroyAllWindows()
	pipeline.stop()
	# pipeline = recording.stop_playback(pipeline)