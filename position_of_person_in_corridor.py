import cv2
import imutils
import numpy as np
import math
import pyrealsense2 as rs
import time

import scipy.spatial

import convert_realsense_to_real_world
import recording
import skeleton_tracker as st
import human_skeleton
# from convert_realsense_to_real_world import *

from recording import *

from cv2 import aruco

import automatic_capture as auto_cap

import InfluxDB_connection


class Map:

    def __init__(self, width_m=3, height_m=3, center_x=None, center_y=None, width_px=600, height_px=600,
                 marker_size=1.0, marker_position="wall"):
        self.names = []
        self.width_m = width_m
        self.width_px = width_px
        self.height_m = height_m
        self.height_px = height_px
        self.scale_x = width_px / width_m
        self.scale_y = height_px / height_m
        self.scale = min(self.scale_x, self.scale_y)
        self.center_x = center_x if center_x is not None else width_m / 2
        self.center_y = center_y if center_y is not None else height_m / 2

        self.reference_2D = None  # Position of the marker in the frame
        self.reference_3D = None  # Position of the marker in 3D (from the camera)

        self.users_position_2D = []  # Position of the users in the frame
        self.users_position_real_world = []  # Position of the users in the real world
        self.users_names = []
        self.users_colors = []

        self.user_walk_tracking = {}
        self.user_walk_tracking_name = {}

        self.user_orientation = []
        self.skeletons = []
        self.previous_skeletons = []

        self.image = None

        self.marker_size = marker_size
        self.marker_position = marker_position

        self.display_axes = [1, 2, 0]
        if marker_position == "ground":
            self.display_axes = [0, 1, 2]

        self.right_footstep_img = cv2.imread("footsteps/right_foot.png", cv2.IMREAD_UNCHANGED)
        self.left_footstep_img = cv2.imread("footsteps/left_foot.png", cv2.IMREAD_UNCHANGED)

    def update_reference_coord(self, marker_corners, depth_frame, pipeline):
        (top_left, top_right, bottom_right, bottom_left) = (
            marker_corners[0][0], marker_corners[0][1], marker_corners[0][2], marker_corners[0][3])
        aruco_origin_3D = convert_realsense_to_real_world.get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
        top_right_3D = convert_realsense_to_real_world.get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
        bottom_left_3D = convert_realsense_to_real_world.get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

        if np.isnan(aruco_origin_3D).any() or np.isnan(top_right_3D).any() or np.isnan(bottom_left_3D).any():
            return None
        else:
            self.reference_2D = {'origin': top_left, 'x_axis': top_right, 'y_axis': bottom_left}
            self.reference_3D = {'origin': aruco_origin_3D, 'x_axis': top_right_3D, 'y_axis': bottom_left_3D}
            return self.reference_3D

    def add_human(self, human, name=None):

        # if marker_corners is not None:
        # 	self.update_reference_coord(marker_corners, depth_frame, pipeline)

        position_real_world = np.array([0, 0, 0])
        self.names.append(name)
        self.skeletons.append(human)

    def display_map(self, bg=(255, 255, 255)):
        self.image = self._init_map(bg=bg)
        self.image = self._add_center_reference()
        # self.image = self._display_users(cross_size=6)
        self.image = self._display_footsteps()
        return self.image

    def _add_center_reference(self):
        if self.reference_2D is not None:
            ref_pos_x = self.center_x * self.scale
            ref_pos_y = self.center_y * self.scale
            self.image = draw_arrow(self.image, (ref_pos_x, ref_pos_y), (ref_pos_x + self.scale, ref_pos_y),
                                    color=(0, 0, 255))
            self.image = draw_arrow(self.image, (ref_pos_x, ref_pos_y), (ref_pos_x, ref_pos_y + self.scale),
                                    color=(255, 0, 0))
        else:
            pass
        return self.image

    def _display_footsteps(self):
        for skeleton_id, skeleton in enumerate(self.skeletons):
            right_foot = skeleton.get_keypoint3D('RIGHT_ANKLE')
            left_foot = skeleton.get_keypoint3D('LEFT_ANKLE')
            neck = skeleton.get_keypoint3D('NECK')
            orientation = skeleton.get_body_orientation()
            orientation_degrees = 0.0

            if orientation is not None:  # Get the orientation to rotate footsteps
                orientation = orientation[self.display_axes[:2]]
                orientation_degrees = math.degrees(np.arccos(np.dot(orientation, np.array([-1, 0])))) * \
                                      (-1 if np.dot(orientation, np.array([0, -1])) < 0 else 1) \
                                      - 90

            if right_foot is not None:
                right_foot = (right_foot[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
                self.image = add_image(self.image, self.right_footstep_img, right_foot, angle=orientation_degrees)

            if left_foot is not None:
                left_foot = (left_foot[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
                self.image = add_image(self.image, self.left_footstep_img, left_foot, angle=orientation_degrees)

            if orientation is not None and neck is not None:
                neck2D = (neck[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
                self.image = draw_arrow(self.image, neck2D, neck2D + orientation * 100, color=(0, 255, 255))

        return self.image

    def clear_users(self):
        self.users_position_real_world = []
        self.users_position_2D = []
        self.users_names = []
        self.user_orientation = []
        self.previous_skeletons = self.skeletons
        self.skeletons = []

    def _init_map(self, bg=(0, 0, 0)):
        self.image = np.zeros((self.height_px, self.width_px, 3), np.uint8)
        self.image[:, :] = bg
        return self.image


def draw_arrow(image, start, end, color=(255, 255, 255), thickness=2):
    # Main stroke
    image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, thickness)

    # Head
    head_angle_deg = math.radians(30)
    head_size = 0.2
    # left
    head_to_tail = np.array([end[0] - start[0], end[1] - start[1]]) * head_size
    head_1_x = head_to_tail[0] * math.cos(head_angle_deg) - head_to_tail[1] * math.sin(head_angle_deg)
    head_1_y = head_to_tail[0] * math.sin(head_angle_deg) + head_to_tail[1] * math.cos(head_angle_deg)
    # right
    head_2_x = head_to_tail[0] * math.cos(-head_angle_deg) - head_to_tail[1] * math.sin(-head_angle_deg)
    head_2_y = head_to_tail[0] * math.sin(-head_angle_deg) + head_to_tail[1] * math.cos(-head_angle_deg)

    image = cv2.line(image, (int(end[0]), int(end[1])), (int(end[0] - head_1_x), int(end[1] - head_1_y)), color,
                     thickness)
    image = cv2.line(image, (int(end[0]), int(end[1])), (int(end[0] - head_2_x), int(end[1] - head_2_y)), color,
                     thickness)
    return image


def draw_cross(image, position, cross_size=3, color=(0, 0, 0), thickness=2):
    image = cv2.line(image, (int(position[0] - cross_size / 2), int(position[1] - cross_size / 2)),
                     (int(position[0] + cross_size / 2), int(position[1] + cross_size / 2)), color, thickness)
    image = cv2.line(image, (int(position[0] - cross_size / 2), int(position[1] + cross_size / 2)),
                     (int(position[0] + cross_size / 2), int(position[1] - cross_size / 2)), color, thickness)
    return image


def add_image(image, new_image, new_image_coord, alpha=1.0, image_position="center", angle=0.0):
    # new_image_coord = np.array(new_image_coord) # Make a copy
    image_rotated = imutils.rotate_bound(new_image, angle)
    new_image_coord = np.array([new_image_coord[1], new_image_coord[0]])
    if image_position == "center":
        new_image_coord -= np.array(image_rotated.shape)[:2] / 2

    left = int((new_image_coord[1]))
    top = int((new_image_coord[0]))
    bottom = int((new_image_coord[0] + image_rotated.shape[0]))
    right = int((new_image_coord[1] + image_rotated.shape[1]))

    image[top:bottom, left:right] = (image_rotated[:, :, :3] * alpha * (image_rotated[:, :, 3, None] / 255)) + (
            image[top:bottom, left:right] * (1 - (alpha * (image_rotated[:, :, 3, None] / 255))))
    return image



if __name__ == "__main__":

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    pipeline = rs.pipeline()
    config = rs.config()
    recording.configure_streams(config)
    aruco_origin = [0, 0, 0]
    marker_size = convert_realsense_to_real_world.marker_size

    skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline=pipeline)
    joint_confidence = 0.2
    pipeline.start(config)
    # skeleton_db2D_and_skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints("recordings/walking_normal.bag")
    # pipeline = start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = True, repeat_playback = True)
    # automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline, recording_dir = "test_record", latency = 10)

    marker_corners = None

    maraudeur = Map(10, 10, center_x=5, center_y=0, marker_size=marker_size,
                    marker_position="camera")  # marker_position = {wall | ground | camera}

    previous_right = [0, 0, 0]
    previous_left = [0, 0, 0]

    align = rs.align(rs.stream.color)

    playing = True
    while True:
        if playing:
            # start_time = time.time()
            frames = pipeline.wait_for_frames()
            frames.keep()
            # frames = align.process(frames)
            color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth_frame = frames.get_depth_frame()

            skeletons = skeletrack.track_skeletons(color_frame, debug_time=False)  # "Live" solution
            # skeletons = skeleton_db2D_and_skeleton_db3D		# "InfluxDB" solution

            marked_frame, corners, ids = convert_realsense_to_real_world.detectMarker(color_frame, aruco_dict)

            if corners:
                marker_corners = corners[0]

            maraudeur.clear_users()

            number_of_persons = skeletrack.get_number_of_skeletons()
            # number_of_persons = len(skeleton_db2D)
            for skeleton_id in range(number_of_persons):
                human = human_skeleton.HumanSkeleton()
                human.from_tracker(skeletrack, skeleton_id, depth_frame)

                maraudeur.add_human(human)

            cv2.imshow("Maraudeur", maraudeur.display_map())
            cv2.imshow("Camera", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Camera", cv2.cvtColor(skeletrack.draw_skeleton(color_frame), cv2.COLOR_BGR2RGB))
        # print("Total     time : ", round((time.time() - start_time), 3))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            pipeline = stop_playback(pipeline)
            break
        if key == ord('p'):
            if playing:
                pipeline = pause_playback(pipeline)
            else:
                pipeline = resume_playback(pipeline)
            playing = not playing
