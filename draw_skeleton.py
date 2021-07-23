"""! In this script we created a small demonstrator that detects a person's skeleton, draw a box around her
and tries to identify the person with face recognition.

@see person_identification.py for the functions used in here
"""

import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
# import colorsys
import time
import realsense_filters as filters

import recording

import human_skeleton
import InfluxDB_connection

import person_identification
import threading

from typing import Dict, List, Union, Tuple

def detect_face(human: human_skeleton.HumanSkeleton, probabilities_good_face: Dict,
                number_of_faces_detection: Dict, color_frame: np.ndarray, skeleton_id: int,
                faces_encodings: Dict = None, classifier = None) -> Tuple[float, int]:
    """! Function that tries to recognize a person on an image and update an array of probability of the person.
    @param human The HumanSkeleton object of the person to identify.
    @param probabilities_good_face A dictionary for this person containing the list of possible faces recognized
    @param number_of_faces_detection A Dictionary containing a counter on the number of times this person has
    been evaluated. This is used to do the statistics at the end.
    @param color_frame The current frame to be analysed
    @param skeleton_id The ID of the person to be identified
    @param faces_encodings The list of encodings created from the faces dataset used to recognize a person.
    This can be omitted if the classifier is given.
    @param classifier A SVM classifier (or whatever classifier) that is used to recognize a person.

    @return A tuple containing the recognition probability and the index of the face recognized (most
    resembling encoding from the "face_encoding" array or the result of the "classifier")
    """
    face_reco = person_identification.recognize_person(human, color_frame, faces_encodings, classifier)
    if face_reco is None:
        return
    probabilities, best_match = face_reco
    # face_distance = face_distance[0]
    if skeleton_id not in probabilities_good_face:
        probabilities_good_face[skeleton_id] = probabilities  # [best_match]
        number_of_faces_detection[skeleton_id] = 1
    else:
        probabilities_good_face[skeleton_id] += probabilities  # [best_match]
        number_of_faces_detection[skeleton_id] += 1
    return face_reco

def main():
    """! Displays a bounding box around a person and tries to identify the person.

    We use a background substation technic to draw the box behind and in front of a person to give an illusion of 3D.
    """
    # from moviepy.editor import concatenate_videoclips, ImageClip
    import argparse

    parser = argparse.ArgumentParser(description='Display the skeleton of a person on the screen and tries to'
                                                 'identify him/her using facial recognition. Press \'Q\' to exit.')
    parser.add_argument('--live', dest='live', help='Do it in real-time, using a depth camera plugged in the computer',
                        action='store_true', default=True)
    parser.add_argument('--file', dest='file', help='Play a sequence from a rosbag file', type=str)

    args = parser.parse_args()
    file = args.file
    # If no file is provided, do it live
    live = file is None

    if live:
        # Configure the RealSense pipeline from the camera to the computer
        pipeline = rs.pipeline()
        config = rs.config()
        recording.configure_streams(config)
        # Start recording
        pipeline.start(config)

    # Background subtract is used to have the background/foreground (display use only)
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500)

    # Images of faces used for recognition
    images = [
        ['faces/Eric-Benoit.jpg'],
        ['faces/Marc-Hartley.jpg', 'faces/marc_face.jpg'],
        ['faces/Stephane-Perrin.jpg'],
        ['faces/Alexandre-Benoit.jpg'],
    ]
    labels = [
        'Eric',
        'Marc',
        'Stéphane',
        'Alexandre',
    ]
    # Use the face encoding DB if you don't want a classifier (1 label = 1 image)
    # faces_encodings = person_identification.create_encodings(images, "faces/all_encodings.encod", verbose = True)
    # faces_encodings = person_identification.get_encodings("faces/all_encodings.encod")
    faces_encodings = None

    # Use the classifier if you want the use of SVM (1 label = multiple images) => More efficient
    classifier = person_identification.train_classifier_for_person_identification(images, "models/svc.classif")
    # classifier = person_identification.get_classifier_for_person_identification("models/svc.classif")
    # classifier = None

    found_persons_names = {}
    probabilities_good_face = {}
    faces_locations = []
    faces_distances = np.array([])
    number_of_faces_detection = {}

    skeleton_db2D = None
    skeleton_db3D = None
    skeletrack = None
    if live:
        skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline=pipeline)
    else:
        # This is used if you want to use a Rosbag instead of the "live" mode
        filename = file
        pipeline = recording.start_playback(filename, repeat_playback = True, realtime = False)
        skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = True).\
            retrieve_skeleton_keypoints(filename)

    start_time = time.time()
    num_frames = 0
    human = None
    playing = True

    recognition_thread = None

    # frames_list = []

    while True:
        if playing:
            try:
                # Get the current frame
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                # If there is an error, it's because it's the end of your rosbag file
                print(e)
                break
            num_frames += 1

            # Get the frame and the depth map
            color_frame = frames.get_color_frame()
            depth_frame = filters.l515_filter(frames.get_depth_frame()).as_depth_frame()
            current_timestamp = recording.get_frame_timestamp(frames)

            color_frame = np.asanyarray(color_frame.get_data()).copy()
            # color_frame = cv2.flip(color_frame, 0)

            # Update the background substractor
            mask = backSub.apply(color_frame)

            original = color_frame.copy()
            depth_image = np.array(rs.colorizer().colorize(
                # frames.get_depth_frame()).get_data())
                filters.l515_filter(frames.get_depth_frame())).get_data())
            try:
                if live:
                    skeletons = skeletrack.track_skeletons(original)
                    number_of_persons = skeletrack.get_number_of_skeletons(original)
                else:
                    number_of_persons = len(skeleton_db2D)

                for skeleton_id in range(number_of_persons):
                    human = human_skeleton.HumanSkeleton()
                    if live:
                        human.from_tracker(skeletrack, skeleton_id, depth_frame)
                    else:
                        human.from_json(skeleton_db2D, skeleton_db3D, current_timestamp, skeleton_id, depth_frame)
                    positions = []
                    for keypoint in human_skeleton.BodyParts:
                        if human.get_keypoint3D(keypoint) is not None:
                            positions.append(human.get_keypoint3D(keypoint))

                    # Take a rough center of the person by averaging all it's keypoints
                    position = np.median(positions, axis=0)

                    # Detect person's face
                    if recognition_thread is None or not recognition_thread.is_alive():
                        recognition_thread = threading.Thread(target=detect_face, kwargs={
                            # detect_face(**{
                            'human': human,
                            'probabilities_good_face': probabilities_good_face,
                            'number_of_faces_detection': number_of_faces_detection,
                            'color_frame': color_frame.copy(),
                            'faces_encodings': faces_encodings,
                            'classifier': classifier,
                            'skeleton_id': skeleton_id,
                        })
                        recognition_thread.start()
                        pass

                    # Take the most probable label for the person
                    if skeleton_id in probabilities_good_face:
                        found_persons_names[skeleton_id] = labels[np.argmax(probabilities_good_face[skeleton_id])]

                    # Person name placed at it's center if possible
                    text_position = (-1000, -1000)
                    if human.get_center_of_gravity_2d() is not None:
                        text_position = tuple(map(int, human.get_center_of_gravity_2d()))

                    # Display the back of the bounding box on the background
                    color_frame = human.display_BBox(color_frame, pipeline, color=(255, 255, 0))
                    # Re-print the foreground (the person)
                    color_frame[mask > 127] = original[mask > 127]
                    color_frame = skeletrack.draw_skeleton(color_frame)
                    # Display the font of the bounding box over the person
                    color_frame = human.display_BBox(color_frame, pipeline, color=(255, 255, 0), display_back=False)

                    # If available, display the person's name
                    if skeleton_id in found_persons_names:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_size = 0.6
                        text_thickness = 2
                        recognized_person_index = np.argmax(probabilities_good_face[skeleton_id])
                        cumm_proba = probabilities_good_face[skeleton_id][recognized_person_index]
                        proba = cumm_proba / number_of_faces_detection[skeleton_id]
                        text = found_persons_names[skeleton_id] + f"({round(proba * 100)}%)"
                        text_width, text_height = cv2.getTextSize(text, font, font_size, text_thickness)[0]

                        color_frame = cv2.putText(color_frame,
                                                  text,
                                                  (int(text_position[0] - text_width / 2),
                                                   int(text_position[1] - text_height / 2)),
                                                  font, font_size, (255, 255, 0), text_thickness)
            except Exception as e:
                print(e)
                raise e

            cv2.imshow("Camera", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
            # frames_list.append(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
            # cv2.imshow("Camera", cv2.resize(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB), (800, 600)))

        k = cv2.waitKey(1) & 0xFF
        # Press 'q' or 'ESC' to quit
        if k == ord('q') or k == 27:
            break
        elif k == ord('p'):
            if playing:
                pipeline = recording.pause_playback(pipeline)
            else:
                pipeline = recording.resume_playback(pipeline)
            playing = not playing

    cv2.destroyAllWindows()
    pipeline = recording.stop_playback(pipeline)
    # video_clip = concatenate_videoclips([ImageClip(m).set_duration(1 / 20) for m in frames_list])
    # video_clip.write_videofile('passage2.mp4', fps = 20)


if __name__ == "__main__":
    main()
