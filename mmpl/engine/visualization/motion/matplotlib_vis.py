from . import camera

import cv2
import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from matplotlib.animation import FuncAnimation, writers


def vis_2d_keypoints(
    keypoints, img, skeleton, kp_thresh,
    alpha=0.7, output_file=None, show_name=False):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, skeleton.keypoint_num)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    mask = img.copy()
    root = skeleton.root
    stack = [root]
    while stack:
        parent = stack.pop()
        p_idx = skeleton.keypoint2index[parent]
        p_pos = int(keypoints[p_idx, 0]), int(keypoints[p_idx, 1])
        p_score = keypoints[p_idx, 2] if kp_thresh is not None else None
        if kp_thresh is None or p_score > kp_thresh:
            cv2.circle(
                mask, p_pos, radius=3,
                color=colors[p_idx], thickness=-1, lineType=cv2.LINE_AA)
            if show_name:
                cv2.putText(mask, parent, p_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
        for child in skeleton.children[parent]:
            if child not in skeleton.keypoint2index or \
              skeleton.keypoint2index[child] < 0:
                continue
            stack.append(child)
            c_idx = skeleton.keypoint2index[child]
            c_pos = int(keypoints[c_idx, 0]), int(keypoints[c_idx, 1])
            c_score = keypoints[c_idx, 2] if kp_thresh else None
            if kp_thresh is None or \
              (p_score > kp_thresh and c_score > kp_thresh):
                cv2.line(
                    mask, p_pos, c_pos,
                    color=colors[c_idx], thickness=2, lineType=cv2.LINE_AA)

    vis_result = cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)
    if output_file:
        file = Path(output_file)
        if not file.parent.exists():
            os.makedirs(file.parent)
        cv2.imwrite(str(output_file), vis_result)

    return vis_result


def vis_3d_keypoints( keypoints, skeleton, azimuth, elev=15): 
    x_max, x_min = np.max(keypoints[:, 0]), np.min(keypoints[:, 0])
    y_max, y_min = np.max(keypoints[:, 1]), np.min(keypoints[:, 1])
    z_max, z_min = np.max(keypoints[:, 2]), np.min(keypoints[:, 2])
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azimuth)
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.set_zlim3d([0, 2 * radius])

    root = skeleton.root
    stack = [root]
    while stack:
        parent = stack.pop()
        p_idx = skeleton.keypoint2index[parent]
        p_pos = keypoints[p_idx]
        for child in skeleton.children[parent]:
            if skeleton.keypoint2index.get(child, -1) == -1:
                continue
            stack.append(child)
            c_idx = skeleton.keypoint2index[child]
            c_pos = keypoints[c_idx]
            if child in skeleton.left_joints:
                color = 'b'
            elif child in skeleton.right_joints:
                color = 'r'
            else:
                color = 'k'
            line = ax.plot(
                xs=[p_pos[0], c_pos[0]],
                ys=[p_pos[1], c_pos[1]],
                zs=[p_pos[2], c_pos[2]],
                c=color, marker='.', zdir='z'
            )

    return


def vis_3d_keypoints_sequence(
    keypoints_sequence, skeleton, azimuth,
    fps=30, elev=15, output_file=None
):
    kps_sequence = keypoints_sequence
    x_max, x_min = np.max(kps_sequence[:, :, 0]), np.min(kps_sequence[:, :, 0])
    y_max, y_min = np.max(kps_sequence[:, :, 1]), np.min(kps_sequence[:, :, 1])
    z_max, z_min = np.max(kps_sequence[:, :, 2]), np.min(kps_sequence[:, :, 2])
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azimuth)
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.set_zlim3d([0, 2 * radius])

    initialized = False
    lines = []

    def update(frame):
        nonlocal initialized

        if not initialized:
            root = skeleton.root
            stack = [root]
            while stack:
                parent = stack.pop()
                p_idx = skeleton.keypoint2index[parent]
                p_pos = kps_sequence[0, p_idx]
                for child in skeleton.children[parent]:
                    if skeleton.keypoint2index.get(child, -1) == -1:
                        continue
                    stack.append(child)
                    c_idx = skeleton.keypoint2index[child]
                    c_pos = kps_sequence[0, c_idx]
                    if child in skeleton.left_joints:
                        color = 'b'
                    elif child in skeleton.right_joints:
                        color = 'r'
                    else:
                        color = 'k'
                    line = ax.plot(
                        xs=[p_pos[0], c_pos[0]],
                        ys=[p_pos[1], c_pos[1]],
                        zs=[p_pos[2], c_pos[2]],
                        c=color, marker='.', zdir='z'
                    )
                    lines.append(line)
            initialized = True
        else:
            line_idx = 0
            root = skeleton.root
            stack = [root]
            while stack:
                parent = stack.pop()
                p_idx = skeleton.keypoint2index[parent]
                p_pos = kps_sequence[frame, p_idx]
                for child in skeleton.children[parent]:
                    if skeleton.keypoint2index.get(child, -1) == -1:
                        continue
                    stack.append(child)
                    c_idx = skeleton.keypoint2index[child]
                    c_pos = kps_sequence[frame, c_idx]
                    if child in skeleton.left_joints:
                        color = 'b'
                    elif child in skeleton.right_joints:
                        color = 'r'
                    else:
                        color = 'k'
                    lines[line_idx][0].set_xdata([p_pos[0], c_pos[0]])
                    lines[line_idx][0].set_ydata([p_pos[1], c_pos[1]])
                    lines[line_idx][0].set_3d_properties( [p_pos[2], c_pos[2]]) 
                    line_idx += 1

    anim = FuncAnimation(
        fig=fig, func=update, frames=kps_sequence.shape[0], interval=1000 / fps
    )

    if output_file:
        output_file = Path(output_file)
        if not output_file.parent.exists():
            os.makedirs(output_file.parent)
        if output_file.suffix == '.mp4':
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=3000)
            anim.save(output_file, writer=writer)
        elif output_file.suffix == '.gif':
            anim.save(output_file, dpi=80, writer='imagemagick')
        else:
            raise ValueError(f'Unsupported output format.'
                             f'Only mp4 and gif are supported.')

    return anim
