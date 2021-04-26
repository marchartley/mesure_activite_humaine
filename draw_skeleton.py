import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
# import colorsys
import time
import realsense_filters as filters

# import matplotlib.pyplot as plt

# import convert_realsense_to_real_world as coord_conversion
import recording

import human_skeleton
import InfluxDB_connection

import person_identification
import threading


#### DEMONSTRATOR ####
# Displays a bounding box around a person and tries to identify the person

def detect_face_async(human, probabilities_good_face, number_of_faces_detection, color_frame, faces_encodings=None,
                      classifier=None):
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


if __name__ == "__main__":

    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    pipeline.start(config)

    align_to = rs.stream.depth
    align = rs.align(align_to)

    # Background substractor is used to have the background/foreground (display use only)
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

    # Use the classifier if you want the use of SVM (1 label = multiple images)
    classifier = person_identification.train_classifier_for_person_identification(images, "models/svc.classif")
    # classifier = person_identification.get_classifier_for_person_identification("models/svc.classif")
    # classifier = None

    found_persons_names = {}
    probabilities_good_face = {}
    faces_locations = []
    faces_distances = np.array([])
    number_of_faces_detection = {}

    filename = "recordings/20210416_140741.bag"
    skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline=pipeline)

    # pipeline = recording.start_playback(filename, pipeline = pipeline, repeat_playback = True, realtime = False)
    # skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints(filename)
    joint_confidence = 0.4

    start_time = time.time()
    num_frames = 0
    human = None
    playing = True

    # automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline)

    recognition_thread = None

    while True:
        if playing:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print(e)
                break
            num_frames += 1
            # frames = align.process(frames)
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

                skeletons = skeletrack.track_skeletons(original)
                number_of_persons = skeletrack.get_number_of_skeletons(original)
                # number_of_persons = len(skeleton_db2D)
                for skeleton_id in range(number_of_persons):
                    human = human_skeleton.HumanSkeleton()
                    human.from_tracker(skeletrack, skeleton_id, depth_frame)
                    positions = []
                    for keypoint in human_skeleton.BodyParts:
                        if human.get_keypoint3D(keypoint) is not None:
                            positions.append(human.get_keypoint3D(keypoint))

                    # Take a rough center of the person by averaging all it's keypoints
                    position = np.median(positions, axis=0)

                    # human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id],
                    #                 specific_timestamp = current_timestamp/1000)

                    # if human.check_keypoints(['left_shoulder', 'right_shoulder', 'neck']):
                    # 	left_shoulder = human.get_keypoint3D('left_shoulder')
                    # 	right_shoulder = human.get_keypoint3D('right_shoulder')
                    #
                    # 	left_to_right_shoulders = right_shoulder - left_shoulder
                    # 	neck2D = human.get_keypoint2D('neck')
                    # 	neck3D = human.get_keypoint3D('neck')
                    # 	looking_right = left_to_right_shoulders.dot(neck3D) > 0

                    # body_orientation = human.get_body_orientation()
                    # if body_orientation is not None:
                    # 	body_orientation = (body_orientation / np.linalg.norm(body_orientation)) * 400
                    # 	body_orientation2D = rs.rs2_project_point_to_pixel(pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(), (neck3D + body_orientation).tolist())
                    #
                    # 	color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1])), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
                    # 	if body_orientation[2] > 0:
                    # 		color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
                    # 	else:
                    # 		color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1] + 10)), (0, 255, 255), 2)
                    # 	legs_overlapping = human.legs_overlapping()

                    # Detect person's face
                    if recognition_thread is None or not recognition_thread.is_alive():
                        recognition_thread = threading.Thread(target=detect_face_async, kwargs={
                            # detect_face_async(**{
                            'human': human,
                            'probabilities_good_face': probabilities_good_face,
                            'number_of_faces_detection': number_of_faces_detection,
                            'color_frame': color_frame.copy(),
                            'faces_encodings': faces_encodings,
                            'classifier': classifier,
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
                        text = found_persons_names[
                                   skeleton_id] + f"({round((probabilities_good_face[skeleton_id][np.argmax(probabilities_good_face[skeleton_id])] / number_of_faces_detection[skeleton_id]) * 100)}%)"
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
        # cv2.imshow("Camera", cv2.resize(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB), (800, 600)))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            if playing:
                pipeline = recording.pause_playback(pipeline)
            else:
                pipeline = recording.resume_playback(pipeline)
            playing = not playing

    cv2.destroyAllWindows()
    pipeline = recording.stop_playback(pipeline)
