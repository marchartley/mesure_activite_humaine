import datetime
import os

import dateutil.parser
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import feet_analysis
import human_skeleton
import skeleton_tracker

def visual_skeleton():
    data = feet_analysis.get_data_from_db(-1, time_between_sequences=10.0)
    print(f"Data from {datetime.datetime.fromtimestamp(data['timestamps'][0]).strftime('%Y-%m-%d %H:%M:%S')} "
          f"to {datetime.datetime.fromtimestamp(data['timestamps'][-1]).strftime('%Y-%m-%d %H:%M:%S')}")
    fig = plt.figure()
    ax = Axes3D(fig)

    plt.ion()

    # data normalization

    # show every frame 3d skeleton
    min_max_x = [0, 0]
    min_max_y = [0, 0]
    min_max_z = [0, 0]
    for frame_idx in range(len(data['frames'])):

        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        ax.set_xlim3d([-2, 1])
        ax.set_zlim3d([-1, 3])
        ax.set_ylim3d([-2, 7])

        human = human_skeleton.HumanSkeleton()
        human.from_feet_analytics(data['coordinates'], frame_idx)
        for edge in skeleton_tracker.edges_for_skeleton_drawing:
            if human.check_keypoints3D(edge) and data['velocities'][edge[0]][frame_idx] > 0 and data['velocities'][edge[1]][frame_idx] > 0:
                start_x, start_y, start_z = human.get_keypoint3D(edge[0])
                end_x, end_y, end_z = human.get_keypoint3D(edge[1])
                min_max_x = [min(min_max_x[0], start_x, end_x), max(min_max_x[1], start_x, end_x)]
                min_max_y = [min(min_max_y[0], start_y, end_y), max(min_max_y[1], start_y, end_y)]
                min_max_z = [min(min_max_z[0], start_z, end_z), max(min_max_z[1], start_z, end_z)]
                ax.plot([start_x, end_x], [start_z, end_z], [start_y, end_y], color='b', marker=None)
        # for keypoint in human.body_parts_3D:
        #     if human.body_parts_3D[keypoint] is not None:
        #         x_plot, y_plot, z_plot = human.body_parts_3D[keypoint]
        #         ax.scatter([x_plot], [z_plot], [y_plot], color='b', marker='o')  # , markerfacecolor='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_facecolor('none')
        plt.pause(0.1)
    # print('x:', min_max_x)
    # print('y:', min_max_y)
    # print('z:', min_max_z)
    plt.ioff()
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    visual_skeleton()
