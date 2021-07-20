"""! This small script enables us to display the skeleton in 3D of the last walking stored in the database.

With this script you will discover that the Aruco box doesn't align things very well... A straight walk
in the corridor will result in a small deviation on the X-axis.
Remember that if you want to distinguish walks on the "West", "Center" and "East", that you cannot separate only with
the X-axis position of the person, but you might have to think about a small equation or, if motivated, using a SVM.
"""

import datetime
import os

import dateutil.parser
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import feet_analysis
import human_skeleton
import skeleton_tracker

def visual_skeleton(sequence_id = -1):
    """! Displays the sequence of a skeleton in a 3D plot.
    @param sequence_id The ID of the sequence you want to display
    """
    # Get the data from the database
    data = feet_analysis.get_data_from_db(sequence_id, time_between_sequences=1.0,
                                          server_side = True,
                                          start_time=1619395200000,
                                          ignore_small_sequences=0.2)
    print(data)
    print(f"Data from {datetime.datetime.fromtimestamp(data['timestamps'][0]).strftime('%Y-%m-%d %H:%M:%S')} "
          f"to {datetime.datetime.fromtimestamp(data['timestamps'][-1]).strftime('%Y-%m-%d %H:%M:%S')}")
    fig = plt.figure()
    ax = Axes3D(fig)

    # Activate the interactive mode of the plots
    plt.ion()

    # Store the mean position of the person
    mean_pos = []
    percent_stripped = 0.2

    # show every frame 3d skeleton
    for frame_idx in range(len(data['frames'])):
        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        ax.set_xlim3d([-2, 7])  # ([-2, 1])
        ax.set_zlim3d([-1, 3])
        ax.set_ylim3d([-2, 7])

        # Extract the current skeleton
        human = human_skeleton.HumanSkeleton()
        human.from_feet_analytics(data['coordinates'], frame_idx)
        # Draw each member of the body
        for edge in skeleton_tracker.edges_for_skeleton_drawing:
            # Check if the two joints of the member part are in sight
            if human.check_keypoints3D(edge) and data['velocities'][edge[0]][frame_idx] > 0 \
                    and data['velocities'][edge[1]][frame_idx] > 0:
                # Draw
                start_x, start_y, start_z = human.get_keypoint3D(edge[0])
                end_x, end_y, end_z = human.get_keypoint3D(edge[1])
                ax.plot([start_x, end_x], [start_z, end_z], [start_y, end_y], color='b', marker=None)

                # Add the positions to the mean position
                mean_pos.append(human.get_keypoint3D(edge[0]))
                mean_pos.append(human.get_keypoint3D(edge[1]))

        # Display the joints as small circles
        for keypoint in human.body_parts_3D:
            if human.body_parts_3D[keypoint] is not None:
                x_plot, y_plot, z_plot = human.body_parts_3D[keypoint]
                ax.scatter([x_plot], [z_plot], [y_plot], color=(0, 0, 1, .2), marker='o', s=4**2)

        # Display the head ~20cm above the neck (could be better computed, but it's already not bad
        if human.check_keypoints3D('neck'):
            x_plot, y_plot, z_plot = human.get_keypoint3D('neck')
            ax.scatter([x_plot], [z_plot], [y_plot + 0.2], color=(1, 0, 0, .5), marker='o', s=15**2)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_facecolor('none')
        plt.pause(0.1)

    # To get the mean position of the person, we strip the beginning and ending as it can be a little biased
    # (like going through the door, or anything)
    stripped_part = round(len(mean_pos) * percent_stripped)
    # And we do the mean with the middle part of the sequence
    print(f"Mean position during sequence : {np.mean(mean_pos[stripped_part:-stripped_part], axis = 0)}")
    plt.ioff()
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    # Display last sequence recorded
    visual_skeleton(sequence_id = -1)
