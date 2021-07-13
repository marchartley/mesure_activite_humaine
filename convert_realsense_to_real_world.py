"""! This file regroup multiple functions that are useful when you want to convert positions from
a referential into another.
For example when using the Aruco box, it allows to easily switch from the 2D coordinate of a pixel into the
3D coordinate relative to the box.
"""

import cv2
import numpy as np
from typing import List, Tuple, Union, Dict
import pyrealsense2 as rs

import human_skeleton
import recording
import skeleton_tracker
from recording import *

from cv2 import aruco

import realsense_filters as filters

## The accepted type of variable for 2D coordinates
Coord2D = Union[List, Tuple, np.ndarray]
## The accepted type of variable for 3D coordinates
Coord3D = Union[List, Tuple, np.ndarray]
## The accepted type of variable for a coordinate with no limit of dimensions
Coord = Union[List, Tuple, np.ndarray]
## The type used when outputing coordinates from the functions in this file. Alias of ndarray.
# @see numpy.ndarray
Output_coord = np.ndarray
## Image type
Image = np.ndarray
## Vector
Vector = Union[np.ndarray, List]

## We know that our camera can see up to 5m, so if it find any distance > 5m
# or lower than 0.5m, it must be a wrong value
forced_cube_distance_range = (0.5, 5.0)  # Box is always between 50cm to 5m from camera, to avoid wrong values

## Size of Aruco markers on the Aruco box (in meters)
marker_size = 0.21

## The AruCo box has 6 markers describing the X, Y and Z axis
# For example, the face with marker '0' is on the plane XY, and the Z-axis is the opposite of the cross product of X and Y
# But the face '1' is on the plane ZY (with the Z having to be inversed) and the X-axis is the cross product of Z and Y
aruco_box = {
    0: {
        'vertical': 'y',
        'horizontal': 'x',
        'inverse_third_dim': True
    },
    1: {
        'vertical': 'y',
        'horizontal': '-z',
        'inverse_third_dim': False
    },
    2: {
        'vertical': 'y',
        'horizontal': '-x',
        'inverse_third_dim': True
    },
    3: {
        'vertical': 'y',
        'horizontal': 'z',
        'inverse_third_dim': False
    },
    4: {
        'vertical': 'z',
        'horizontal': 'x',
        'inverse_third_dim': False
    },
    5: {
        'vertical': 'z',
        'horizontal': '-x',
        'inverse_third_dim': False
    }
}

def generateMarkers(aruco_dict: int, n_x: int = 1, n_y: int = 1) -> List[Image]:
    """! Function that use OpenCV functions to create the Aruco tags
    @param aruco_dict Dictionary identifier given by Aruco
    @param n_x Number of tags to generate in a row
    @param n_y Number of tags to generate in a column

    @return A set of aruco tags
    """
    ## Desired size of the borders of the marker (quite big so it is easier to detect the marker)
    border = 100
    ## Size in pixel of the marker
    size = 300

    tags = []

    aruco_target = np.zeros((border * (n_y + 1) + size * n_y, border * (n_x + 1) + size * n_x))
    aruco_target[::] = 255

    for i in range(n_y * n_x):
        img = aruco.drawMarker(aruco_dict, i, size)
        tags.append(img)
        y_offset = border + (i // n_x) * (size + border)
        x_offset = border + (i % n_x) * (size + border)
        aruco_target[y_offset:y_offset + img.shape[0],
        x_offset:x_offset + img.shape[1]] = img

    return tags

def detectMarker(frame: Image, aruco_dict: dict) -> Tuple[Image, np.ndarray, List[int]]:
    """! Function that use the OpenCV functions to find the Aruco tags on an image
    @param frame The image of the camera on which a marker might be detected
    @param aruco_dict Dictionary identifier given by Aruco

    @return A tuple containing: \n
    - The original image with the marker highlighted (for debug) \n
    - The coordinates of each corner of each marker detected \n
    - The ID of the markers detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize=(
            3, 3), zeroZone=(-1, -1), criteria=criteria)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    return frame_markers, corners, ids


def get_3D_coord_from_realsense(pixel_coord: Coord2D, depth_frame: rs.depth_frame,
                                pipeline: rs.pipeline) -> Output_coord:
    """! Transform a 2D coord from the color frame to the 3D coord given from the camera
    @param pixel_coord The 2D coordinates for which you want to transform in 3D
    @param depth_frame The depth frame provided from the camera
    @param pipeline The RealSense pipeline from the camera

    @return An array with the X, Y, and Z coordinates of the original point
    """
    # Get all parameters from the camera sensors
    depth_stream = pipeline.get_active_profile().get_stream(rs.stream.depth)
    color_stream = pipeline.get_active_profile().get_stream(rs.stream.color)

    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_option(rs.option.depth_units)  # depth_sensor.get_depth_scale()

    depth_stream_profile = depth_stream.as_video_stream_profile()
    color_stream_profile = color_stream.as_video_stream_profile()

    min_depth = 0.01
    max_depth = 16.0

    depth_intrin = depth_stream_profile.get_intrinsics()
    color_intrin = color_stream_profile.get_intrinsics()

    depth_to_color_extrin = depth_stream_profile.get_extrinsics_to(color_stream_profile)
    color_to_depth_extrin = color_stream_profile.get_extrinsics_to(depth_stream_profile)

    pixel_coord = np.array(pixel_coord, int).tolist()

    # Computing the depth this way works well for the D435 and the L515 (not sure how it works, tho)
    color_point = rs.rs2_deproject_pixel_to_point(color_intrin, pixel_coord, 1)
    depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, color_point)
    depth_coord = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)

    # This method is very imprecise using the L515 camera
    # depth_coord = rs.rs2_project_color_pixel_to_depth_pixel(
    #     depth_frame.get_data(), depth_scale,
    #     min_depth, max_depth,
    #     depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, pixel_coord)

    # TODO: This has to be improved to check for negative values and too big values...
    # Ideally, find exactly why it raise exceptions (maybe replace the round function...)
    try:
        distance = depth_frame.get_distance(round(depth_coord[0]), round(depth_coord[1]))
    except:
        distance = 0

    # Finally, return 3D coord
    return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, depth_coord, distance))  # / depth_scale


def back_to_pixel(point_coord: Coord3D, pipeline: rs.pipeline, old_origin_coord: Coord3D = None,
                  old_x_axis: Coord3D = None, old_y_axis: Coord3D = None, old_z_axis: Coord3D = None) -> Output_coord:
    """! Compute the 2D coord from a 3D point. The default origin is the camera, but you can give
    the coords of the Aruco marker if the point was already translated

    @note This uses the RealSense SDK but is still not very precise, it should be used only for debugging usage

    @param point_coord The 3D point that needs to be projected in 2D space
    @param pipeline The RealSense pipeline from the camera
    @param old_origin_coord The 3D coordinates of the origin of the referential for which the 3D point has been
    calculated
    @param old_x_axis The 3D coordinates of the point (1, 0, 0) of the "old" referential
    @param old_y_axis The 3D coordinates of the point (0, 1, 0) of the "old" referential
    @param old_z_axis The 3D coordinates of the point (0, 0, 1) of the "old" referential

    @return The 2D coordinates after projection on the screen from the original point
    """
    if old_origin_coord is None:
        old_origin_coord = [0, 0, 0]
    if old_x_axis is None:
        old_x_axis = [1, 0, 0]
    if old_y_axis is None:
        old_y_axis = [0, 1, 0]
    color_intrin = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # If an origin is given, retransform the point to be in the camera origin
    x_axis = np.array(old_x_axis) - np.array(old_origin_coord)
    y_axis = np.array(old_y_axis) - np.array(old_origin_coord)
    if old_z_axis is None:
        z_axis = compute_z_axis(x_axis, y_axis)
    else:
        z_axis = np.array(old_z_axis) - np.array(old_origin_coord)

    transform = np.matrix([x_axis, y_axis, z_axis])
    coord_from_cam = np.dot(point_coord, transform)
    coord_from_cam += old_origin_coord

    # Get the 2D coord
    return np.array(rs.rs2_project_point_to_pixel(color_intrin, coord_from_cam.tolist()[0]))

def translate_coord_to_world(coord: Coord3D, new_origin_coord: Coord3D, new_x_axis_coord: Coord3D,
                             new_y_axis_coord: Coord3D, new_z_axis_coord: Coord3D=None,
                             scale: float = 1.0) -> np.ndarray:
    """! Compute the 3D coordinates from a point that was in the camera origin to another origin (ex: Aruco box)
    @param coord The 3D coordinates of a point from the camera
    @param new_origin_coord The 3D coordinates of the origin of the new referential
    @param new_x_axis_coord The 3D coordinates of the point (1, 0, 0) in the new referential
    @param new_y_axis_coord The 3D coordinates of the point (0, 1, 0) in the new referential
    @param new_z_axis_coord The 3D coordinates of the point (0, 0, 1) in the new referential
    @param scale The scale of the new referential

    @return The 3D coordinates of the original point in the new referential

    @note The scale is useful if the "new_(x|y|z)_axis_coord" are not at 1m from the origin. For example, the
    Aruco box has markers of size 0.21m, so you can take the corners of the marker as axis coords and apply
    a scale of 0.21 to compensate.
    """
    # Get the starting reference (the camera)
    # which is the identity matrix + nul-vector
    original_X = np.matrix([1, 0, 0]).transpose()
    original_Y = np.matrix([0, 1, 0]).transpose()
    original_Z = np.matrix([0, 0, 1]).transpose()
    M1 = np.column_stack((original_X, original_Y, original_Z))  # It's just the identity matrix
    p1 = np.matrix(coord).transpose()
    O1 = np.matrix([0, 0, 0]).transpose()

    # Compute the new origin and axis from their 3D coordinates
    O2 = np.matrix(new_origin_coord).transpose()
    new_X = np.matrix(new_x_axis_coord).transpose() - O2
    new_Y = np.matrix(new_y_axis_coord).transpose() - O2
    if new_z_axis_coord is not None:
        new_Z = np.matrix(new_z_axis_coord).transpose() - O2
    else:
        new_Z = compute_z_axis(new_X, new_Y)
    M2 = np.column_stack((new_X, new_Y, new_Z))

    # The position of the point is computed by
    # p2 = M2' * (O1 - O2 + M1 * p1)
    # with p1 -> the original point, M1 -> the 3 original axis' coordinates and O1 -> the original origin coordinates
    # and p2 -> the translated point
    p2 = np.dot(np.linalg.inv(M2), ((O1 - O2) + np.dot(M1, p1)))
    # Rescale the result if needed
    return p2 * scale

def compute_z_axis(x_axis: Vector, y_axis: Vector) -> Vector:
    """! Compute a 3rd axis when only 2 are available by using the cross product
    @param x_axis The 3D vector of the X-axis
    @param y_axis The 3D vector of the Y-axis

    @return The 3D vector of the computed Z-axis
    """
    z_axis = np.cross(np.array(x_axis).transpose(), np.array(y_axis).transpose()).transpose()
    # Getting a coherent Z norm by resizing it to the mean of the X vector's norm and the Y vector's norm
    norm_X = np.linalg.norm(x_axis)
    norm_Y = np.linalg.norm(y_axis)
    norm_Z = np.linalg.norm(z_axis)
    # TODO: This is an approximation, if a nice mathematician want to correct me it would be nice!
    z_axis = (z_axis / norm_Z) * ((norm_X + norm_Y) / 2)
    return z_axis


def draw_line(image: Image, start: Coord2D, end: Coord2D,
              color: Union[Tuple, List] = (0, 0, 0),
              thickness: int = 2) -> Image:
    """! Drawing a line on an image
    @param image The image on which you want to draw a line
    @param start Position of the starting point of the line
    @param end Position of the ending point of the line
    @param color Color of the line in the format (R, G, B) and values between 0 and 255
    @param thickness Width of the line, in pixel

    @return A modified copy of the image
    """
    image = cv2.line(image.copy(), tuple(map(int, start)), tuple(map(int, end)), color=color, thickness=thickness)
    return image

def draw_cube(image: Image, marker: List, depth_frame: rs.depth_frame, pipeline: rs.pipeline) -> Image:
    """! Drawing a cube at the marker's location on an image
    @param image The original image
    @param marker The position of the corners of the markers detected. This is found in the output of detectMarker().
    @param depth_frame The depth frame coming from the camera
    @param pipeline The RealSense pipeline from the camera

    @return A modified copy of the image
    """
    (top_left, top_right, bottom_right, bottom_left) = (marker[0][0], marker[0][1], marker[0][2], marker[0][3])

    top_left_3D = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
    top_right_3D = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
    bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

    # If we are missing a coord in the depth map, don't even try to draw the cube
    if np.all(top_left_3D == 0) or np.all(top_right_3D == 0) or np.all(bottom_left_3D == 0):
        return image

    # Compute the 8 coordinates of the cube
    p000 = back_to_pixel([0, 0, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p001 = back_to_pixel([0, 0, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p010 = back_to_pixel([0, 1, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p011 = back_to_pixel([0, 1, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p100 = back_to_pixel([1, 0, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p101 = back_to_pixel([1, 0, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p110 = back_to_pixel([1, 1, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
    p111 = back_to_pixel([1, 1, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)

    ## Display the lines ##
    # Depth
    image = draw_line(image, p000, p001, (0, 255, 0))
    image = draw_line(image, p010, p011, (0, 255, 0))
    image = draw_line(image, p100, p101, (0, 255, 0))
    image = draw_line(image, p110, p111, (0, 255, 0))

    # Vertical
    image = draw_line(image, p000, p010, (255, 0, 0))
    image = draw_line(image, p001, p011, (255, 0, 0))
    image = draw_line(image, p100, p110, (255, 0, 0))
    image = draw_line(image, p101, p111, (255, 0, 0))

    # Horizontal
    image = draw_line(image, p000, p100, (0, 0, 255))
    image = draw_line(image, p001, p101, (0, 0, 255))
    image = draw_line(image, p010, p110, (0, 0, 255))
    image = draw_line(image, p011, p111, (0, 0, 255))

    # Display the borders of the original marker to compare the error created by the 2D-to-3D-to-2D transformation
    image = draw_line(image, top_left, top_right, (0, 0, 255), thickness=1)
    image = draw_line(image, bottom_right, bottom_left, (0, 0, 255), thickness=1)
    image = draw_line(image, top_left, bottom_left, (255, 0, 0), thickness=1)
    image = draw_line(image, top_right, bottom_right, (255, 0, 0), thickness=1)
    return image


def center_of_shape(vertices: List[Coord]) -> Output_coord:
    """! Return the center of gravity of a shape
    @param vertices the list of coordinate points that defines the shape

    @return The center of gravity
    """
    return np.mean(np.array(vertices), axis=0)

def create_aruco_cube(corners: np.ndarray, ids: List[int], depth_frame: rs.depth_frame, pipeline: rs.pipeline) \
        -> Tuple[Output_coord, Dict[str, Output_coord], List[Output_coord],  Dict[str, List[Output_coord]]]:
    """! Function to create a referential from the Aruco markers that can be found on the Aruco box
    @param corners List of the corners from each Aruco marker found in an image.
    This is found in the output of detectMarker().
    @param ids List of the IDs of the markers detected on screen
    @param depth_frame The depth frame coming from the camera
    @param pipeline The RealSense pipeline from the camera

    @return A tuple containing:
    - The center of the new referential
    - A Dict object with the position of each axis {'x': position_x, 'y': position_y, 'z': position_z}
    - The list of each center of markers
    - The list of each axis computed by each markers

    @note
    The Aruco cube is using 6 Aruco tags. The camera can see at max 3 markers at the time,
    we want to get the maximum of precision on the axis by using the average of all markers seen
    """
    vectors_counts = {'x': 0, 'y': 0, 'z': 0}
    reference_vectors = {'x': np.array([0.0, 0.0, 0.0]),
                         'y': np.array([0.0, 0.0, 0.0]),
                         'z': np.array([0.0, 0.0, 0.0])}
    all_reference_vectors = {'x': [],
                             'y': [],
                             'z': []}
    if len(corners) == 0:
        return None, None, None, None

    ids = np.array(ids).flatten()
    centers3D = []
    for i, corner in enumerate(corners):
        # Get the 2D values of the corners of the tag
        (top_left, top_right, bottom_right, bottom_left) = (corner[0][0], corner[0][1], corner[0][2], corner[0][3])
        # Get the center of the shape and compute it's 3D coordinates
        center2D = center_of_shape(corner[0])
        centers3D.append(get_3D_coord_from_realsense(center2D, depth_frame, pipeline))

        top_left_3D = get_3D_coord_from_realsense(top_left, depth_frame, pipeline)
        top_right_3D = get_3D_coord_from_realsense(top_right, depth_frame, pipeline)
        bottom_left_3D = get_3D_coord_from_realsense(bottom_left, depth_frame, pipeline)
        bottom_right_3D = get_3D_coord_from_realsense(bottom_right, depth_frame, pipeline)

        # Check if any point is badly detected
        if forced_cube_distance_range[0] > top_left_3D[2] or forced_cube_distance_range[1] < top_left_3D[2]:
            top_left_3D = np.array([0, 0, 0]) # Consider the value wrong if depth is an outlier
        if forced_cube_distance_range[0] > top_right_3D[2] or forced_cube_distance_range[1] < top_right_3D[2]:
            top_right_3D = np.array([0, 0, 0]) # Consider the value wrong if depth is an outlier
        if forced_cube_distance_range[0] > bottom_left_3D[2] or forced_cube_distance_range[1] < bottom_left_3D[2]:
            bottom_left_3D = np.array([0, 0, 0]) # Consider the value wrong if depth is an outlier
        if forced_cube_distance_range[0] > bottom_right_3D[2] or forced_cube_distance_range[1] < bottom_right_3D[2]:
            bottom_right_3D = np.array([0, 0, 0]) # Consider the value wrong if depth is an outlier

        # Store which corner is badly detected to avoid using it in the computation
        missing_values = np.array([np.any(top_left_3D == 0), np.any(top_right_3D == 0), np.any(bottom_left_3D == 0), np.any(
            bottom_right_3D == 0)])
        # Cancel the use of this shape if there are less than 3 corners OK
        if len(missing_values[missing_values == True]) >= 2:
            continue

        # Get axis associated with this marker
        v_ax = aruco_box[ids[i]]['vertical'][-1]  # last char is "x", "y" or "z"
        v_sign = -1 if aruco_box[ids[i]]['vertical'][0] == "-" else 1
        h_ax = aruco_box[ids[i]]['horizontal'][-1]  # last char is "x", "y" or "z"
        h_sign = -1 if aruco_box[ids[i]]['horizontal'][0] == "-" else 1

        last_ax = 'x' if 'x' not in (v_ax, h_ax) else 'y' if 'y' not in (v_ax, h_ax) else 'z'
        last_sign = -1 if aruco_box[ids[i]]['inverse_third_dim'] else 1

        # Use the mean vector between the horizontal and vertical corners
        horizontal_axis = []
        if not missing_values[0] and not missing_values[1]:
            horizontal_axis.append(top_left_3D - top_right_3D)
        if not missing_values[2] and not missing_values[3]:
            horizontal_axis.append(bottom_left_3D - bottom_right_3D)
        horizontal_axis = np.mean(np.array(horizontal_axis),
                                  axis=0) * h_sign

        vertical_axis = []
        if not missing_values[0] and not missing_values[2]:
            vertical_axis.append(top_left_3D - bottom_left_3D)
        if not missing_values[1] and not missing_values[3]:
            vertical_axis.append(top_right_3D - bottom_right_3D)
        vertical_axis = np.mean(np.array(vertical_axis),
                                axis=0) * v_sign

        # Store these axis values
        reference_vectors[v_ax] += vertical_axis
        reference_vectors[h_ax] += horizontal_axis
        reference_vectors[last_ax] += compute_z_axis(vertical_axis, horizontal_axis) * last_sign

        all_reference_vectors[v_ax].append(vertical_axis)
        all_reference_vectors[h_ax].append(horizontal_axis)
        all_reference_vectors[last_ax].append(compute_z_axis(vertical_axis, horizontal_axis) * last_sign)

        vectors_counts[v_ax] += 1
        vectors_counts[h_ax] += 1
        vectors_counts[last_ax] += 1

    # Take the average fore each axis
    for axis in reference_vectors:
        if vectors_counts[axis] > 0:
            reference_vectors[axis] /= vectors_counts[axis]  # Mean of each axis

    # The center is also translated into the center of the cube
    # The center is not the same direction depending on the face, so we use the information of the 3rd axis
    for i in range(len(centers3D)):
        ax_v = aruco_box[ids[i]]['vertical'][-1]
        ax_h = aruco_box[ids[i]]['horizontal'][-1]

        v_dir = 1 if aruco_box[ids[i]]['vertical'][0] == "-" else 0
        h_dir = 1 if aruco_box[ids[i]]['horizontal'][0] == "-" else 0
        inv3dim = 1 if aruco_box[ids[i]]['inverse_third_dim'] else 0

        # Guess I have to understand why is that, but for now it works
        # If there is exactly 1 "negative" axis, you have to invert the offset
        center_offset = -1 if (v_dir + h_dir + inv3dim) % 2 == 0 else 1

        if 'x' not in (ax_v, ax_h):
            centers3D[i] += reference_vectors['x'] * 0.5 * center_offset
        if 'y' not in (ax_v, ax_h):
            centers3D[i] += reference_vectors['y'] * 0.5 * center_offset
        if 'z' not in (ax_v, ax_h):
            centers3D[i] += reference_vectors['z'] * 0.5 * center_offset

    # Cancel "centers" where the depth is not in the "normal" range.
    centers3D = list(filter(lambda coord: forced_cube_distance_range[0] <= coord[2] <= forced_cube_distance_range[1], centers3D))

    # Take the mean average of all the centers computed
    mean_center = np.mean(np.array(centers3D), axis=0)

    # If there is still an incorrect value in the reference vectors, cancel all
    # NOTE: If any vector has a coordinate with 0 (ex: horizontal or vertical center), it is considered incorrect
    # but this is very unlikely to happen...
    if np.any(reference_vectors['x'] == 0) or np.any(reference_vectors['y'] == 0) or np.any(
            reference_vectors['z'] == 0):
        return None, None, None, None

    return mean_center, reference_vectors, centers3D, all_reference_vectors

## Global variable for the Aruco cube reference
all_references_for_aruco_cube = {
    'x_axis': [],
    'y_axis': [],
    'z_axis': [],
    'center': []
}

def calibrate_aruco_cube(color_frame: Image, depth_frame: rs.depth_frame, pipeline: rs.pipeline,
                         aruco_dict: int = None) -> Tuple[Output_coord, Output_coord, Output_coord, Output_coord]:
    """! Improve the precision of the aruco cube at each frame
    @param color_frame The color frame coming from the camera
    @param depth_frame The depth frame coming from the camera
    @param pipeline The RealSense pipeline for the camera
    @param aruco_dict The dictionary used to create the Aruco markers on the cube

    @return A tuple containing:
    - The center of the referential created from the Aruco box
    - The position of the point (1, 0, 0) from the camera
    - The position of the point (0, 1, 0) from the camera
    - The position of the point (0, 0, 1) from the camera
    """
    if aruco_dict is None:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # Get the Aruco tags from the color frame and compute the axis and reference
    marked_frame, corners, ids = detectMarker(color_frame, aruco_dict)
    if len(corners) > 0:
        try:
            center, vectors, all_centers, all_vectors = create_aruco_cube(corners, ids, depth_frame, pipeline)
            if center is not None:
                for a_center in all_centers:
                    all_references_for_aruco_cube['center'].append(a_center)
                for x_vector in all_vectors['x']:
                    all_references_for_aruco_cube['x_axis'].append(np.array(x_vector))
                for y_vector in all_vectors['y']:
                    all_references_for_aruco_cube['y_axis'].append(np.array(y_vector))
                for z_vector in all_vectors['z']:
                    all_references_for_aruco_cube['z_axis'].append(np.array(z_vector))
        except Exception as e:
            raise(e)

    # The median value is good to filter out outliers
    if len(all_references_for_aruco_cube['center']) > 0:
        median_center = np.median(all_references_for_aruco_cube['center'], axis=0)
        median_x = np.median(all_references_for_aruco_cube['x_axis'], axis=0)
        median_y = np.median(all_references_for_aruco_cube['y_axis'], axis=0)
        median_z = np.median(all_references_for_aruco_cube['z_axis'], axis=0)

        return median_center, median_x, median_y, median_z

    else:
        return None, None, None, None

def get_aruco_reference() -> Tuple[Output_coord, Output_coord, Output_coord, Output_coord]:
    """! Function to get the calibrated reference of the Aruco box"""
    # The median value is good to filter out outliers
    if len(all_references_for_aruco_cube['center']) > 0:
        median_center = np.median(all_references_for_aruco_cube['center'], axis=0)
        median_x = np.median(all_references_for_aruco_cube['x_axis'], axis=0)
        median_y = np.median(all_references_for_aruco_cube['y_axis'], axis=0)
        median_z = np.median(all_references_for_aruco_cube['z_axis'], axis=0)

        return median_center, median_x, median_y, median_z
    else:
        return None, None, None, None

def get_number_of_calibration_of_aruco_box() -> int:
    """! Function to count the number of time the Aruco box has been calibrated"""
    return len(all_references_for_aruco_cube['center'])

def display_reference(color_frame: Image, pipeline: rs.pipeline) -> Image:
    """! Graphic function that display the 3 axis from the aruco box
    @param color_frame The color frame coming from the camera
    @param pipeline The RealSense pipeline for the camera

    @return A transformed copy of the image
    """
    new_color_frame = color_frame.copy()
    median_center, median_x, median_y, median_z = get_aruco_reference()
    if median_center is not None:
        # Display the reference
        center2D = tuple(map(int, back_to_pixel(median_center, pipeline)))
        x_axis = tuple(map(int,
                           back_to_pixel(median_center + median_x * 2, pipeline)))
        y_axis = tuple(map(int,
                           back_to_pixel(median_center + median_y * 2, pipeline)))
        z_axis = tuple(map(int,
                           back_to_pixel(median_center + median_z * 2, pipeline)))
        new_color_frame = cv2.circle(new_color_frame, center2D, 3, (0, 255, 0), -1)
        new_color_frame = cv2.line(new_color_frame, center2D, x_axis, (255, 0, 0), 4)
        new_color_frame = cv2.line(new_color_frame, center2D, y_axis, (0, 0, 255), 4)
        new_color_frame = cv2.line(new_color_frame, center2D, z_axis, (0, 255, 0), 4)
    return new_color_frame

def main():
    """! A demonstration of the use of this file"""
    import argparse

    parser = argparse.ArgumentParser(description='Transform the coordinate reference to fit an Aruco marker')
    parser.add_argument('-f', '--filename', dest='filename', metavar='filename', type=str,
                        help='Play a specific sequence .bag', required=False)
    args = parser.parse_args()
    filename = args.filename

    # The Aruco dictionary must be precised, and it must be the DICT_6x6_250 because that is the one used
    # on the current Aruco cube
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    pipeline = rs.pipeline()
    config = rs.config()
    if filename is None:
        recording.configure_streams(config)
        pipeline.start(config)
    else:
        pipeline = start_playback(filename, pipeline, realtime=True, repeat_playback=True)

    aruco_origin = [0, 0, 0]

    align_to = rs.stream.depth
    align = rs.align(align_to)

    filters.create_custom_filters("custom1", [
        rs.temporal_filter(smooth_alpha=0.060, smooth_delta=50, persistence_control=8)])

    skeletrack = skeleton_tracker.SkeletonTracker(choose_best_module=True, pipeline=pipeline)
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except:
            break

        # frames = align.process(frames)
        color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth_frame = filters.aruco_detection_filters(frames.get_depth_frame()).as_depth_frame()
        # depth_frame = filters.custom_filters(frames.get_depth_frame(), "custom1").as_depth_frame()
        # depth_frame = frames.get_depth_frame()

        # Calibrate if possible the cube and get it's reference
        median_center, median_x, median_y, median_z = calibrate_aruco_cube(color_frame, depth_frame, pipeline,
                                                                           aruco_dict)

        # Track the skeleton for demonstration
        skeleton = skeletrack.track_skeletons(color_frame)
        color_frame, _, _ = detectMarker(color_frame, aruco_dict)

        depth_image = np.array(rs.colorizer(2).colorize(depth_frame).get_data())
        try:
            if median_center is not None:
                # Display the reference
                center2D = tuple(map(int, back_to_pixel(median_center, pipeline)))
                x_axis = tuple(map(int,
                                   back_to_pixel(median_center + median_x * 2, pipeline)))
                y_axis = tuple(map(int,
                                   back_to_pixel(median_center + median_y * 2, pipeline)))
                z_axis = tuple(map(int,
                                   back_to_pixel(median_center + median_z * 2, pipeline)))
                color_frame = cv2.circle(color_frame, center2D, 3, (0, 255, 0), -1)
                color_frame = cv2.line(color_frame, center2D, x_axis, (255, 0, 0), 4)
                color_frame = cv2.line(color_frame, center2D, y_axis, (0, 0, 255), 4)
                color_frame = cv2.line(color_frame, center2D, z_axis, (0, 255, 0), 4)

        except Exception as e:
            print(e)

        # If a skeleton is detected, you can get the coordinates of the person relative to the box
        if len(skeleton['body']) > 0 and median_center is not None:
            human = human_skeleton.HumanSkeleton()
            human.from_tracker(skeletrack, 0, depth_frame)
            if human.check_keypoints("LEFT_WRIST"):
                median_center = np.median(all_references_for_aruco_cube['center'], axis=0)
                coord_neck = translate_coord_to_world(human.get_keypoint3D("LEFT_WRIST"),
                                                      median_center,
                                                      median_center + np.median(all_references_for_aruco_cube['x_axis'], axis=0),
                                                      median_center + np.median(all_references_for_aruco_cube['y_axis'], axis=0),
                                                      median_center + np.median(all_references_for_aruco_cube['z_axis'], axis=0),
                                                      marker_size)
                print(coord_neck)

        color_frame = cv2.resize(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB), (800, 600))
        cv2.imshow("color", color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if filename is None:
        pipeline = pipeline.stop()
    else:
        pipeline = stop_playback(pipeline)


if __name__ == "__main__":
    # Example of usage: python convert_realsense_to_real_world.py --file recordings/aruco_code.bag
    main()
