from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import os
import io
import cv2
import imageio
import numpy as np
import tqdm
import torch
from .utils_bvh import quat_fk

def plt2numpy(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def plot_trace(out, interp, R):
    plt.ion()
    fig = plt.figure(0, figsize=(4, 3))
    ax = fig.gca(projection='3d')
    for i in range(1):
        ax.plot(out[i,:,0].cpu(), out[i,:, 2].cpu(), out[i,:, 1].cpu(), 'r')
        ax.plot(R[i,:,0].cpu(), R[i,:,2].cpu(), R[i,:,1].cpu(), 'g')
        ax.plot(interp[i,:,0].cpu(), interp[i,:, 2].cpu(), interp[i,:, 1].cpu(), 'b')
    # ax.set_xlim3d(-100, 100)
    # ax.set_ylim3d(-100, 100)
    # ax.set_zlim3d(0, 200)
    cvs = plt2numpy(fig)
    plt.clf()
    plt.ioff()
    return cvs



def draw_bvh(X, Q, gif_path, mask, tag, parents, duration=0.04):
    cvss = []
    plt.ion()
    if Q is None:
        frames = X
    else:
        _, frames = quat_fk(Q, X, parents)
    for i, pred in tqdm.tqdm(zip(range(len(frames)), mask), mininterval=10):
        if pred:
            color = 'red'
        else:
            color = 'black'
        if pred == 2:
            _alpha = 0.1
        else:
            _alpha = 1
        points = np.array(frames[i])
        points = points[:, [0, 2, 1]]
        fig = plt.figure(0, figsize=(4, 3))
        ax = fig.gca(projection='3d')
        # ax.set_aspect('equal')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=_alpha, linewidths=0, marker='o')
        
        for k in range(len(parents)):
            if parents[k] == -1:
                continue

            start_id = k
            end_id = int(parents[k])
            ax.plot(points[[start_id, end_id], 0],
                    points[[start_id, end_id], 1],
                    points[[start_id, end_id], 2], color=color, alpha=_alpha, linewidth=1)
        
        if i > 1:
            ax.plot(frames[:i, 0, 0],
                    frames[:i, 0, 2],
                    frames[:i, 0, 1], color="green", alpha=0.5, linewidth=0.5)
        
        ax.set_xlim3d(-100, 100)
        ax.set_ylim3d(-100, 100)
        ax.set_zlim3d(0, 200)
        """
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(-200, 200)
        """
        plt.title('{}-{:03d}'.format(tag, i))
        cvs = plt2numpy(fig)
        plt.clf()
        cvss.append(cvs.astype("uint8"))
    if gif_path is not None:
        imageio.mimsave(gif_path, cvss, duration=duration)
    else:
        return cvss


def draw_bvh_pair(pred, interp, gt, gif_path, mask, parents, duration=0.04):
    pred_gif = draw_bvh(pred[0], pred[1], None, mask=mask, parents=parents, tag="pred")
    interp_gif = draw_bvh(interp[0], interp[1], None, mask=mask, parents=parents, tag="interp")
    gt_gif = draw_bvh(gt[0], gt[1], None, mask=mask, parents=parents, tag="gt")
    cvss = [np.hstack([gt, interp, pred]) for gt, interp, pred in zip(pred_gif, interp_gif, gt_gif)]
    imageio.mimsave(gif_path, cvss, duration=duration)


def draw_pair_mp4(pred, interp, gt, mp4_path, mask, fps=30):
    pred_gif = draw(pred, None, mask=mask, tag="pred")
    interp_gif = draw(interp, None, mask=mask, tag="interp")
    gt_gif = draw(gt, None, mask=mask, tag="gt")
    cvss = [np.hstack([gt, interp, pred]) for gt, interp, pred in zip(pred_gif, interp_gif, gt_gif)]
    writer = imageio.get_writer(mp4_path, fps=fps)
    for cvs in cvss:
        writer.append_data(cvs)
    writer.close()


def draw_motion_based_global_pos(
        g_pos,
        parents,
        title='',
        axis_order=[0, 1, 2],
        point_alpha=0.7,
        fg_color='red',
        fg_alpha=0.8,
        gif_path=None,
        duration=0.03
):
    cvss = []
    plt.ion()
    if torch.is_tensor(g_pos):
        g_pos = g_pos.cpu().numpy()  # T x N x 3
    # 如果需要换轴
    g_pos = g_pos[..., axis_order]
    MINS = g_pos.min(axis=0).min(axis=0)
    MAXS = g_pos.max(axis=0).max(axis=0)

    for i in range(len(g_pos)):
        points = g_pos[i]
        fig = plt.figure(0, figsize=(4, 3))
        ax = fig.add_subplot(projection='3d')
        # ax.set_aspect('equal')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=point_alpha, linewidths=0, marker='o')

        if isinstance(parents[0], list):
            for chain in parents:
                ax.plot(points[chain, 0],
                        points[chain, 1],
                        points[chain, 2], color=fg_color, alpha=fg_alpha, linewidth=1)
        else:
            for k in range(len(parents)):
                if parents[k] == -1:
                    continue

                start_id = k
                end_id = int(parents[k])
                ax.plot(points[[start_id, end_id], 0],
                        points[[start_id, end_id], 1],
                        points[[start_id, end_id], 2], color=fg_color, alpha=fg_alpha, linewidth=1)

        if i > 1:
            ax.plot(g_pos[:i, 0, 0],
                    g_pos[:i, 0, 2],
                    g_pos[:i, 0, 1], color="green", alpha=0.5, linewidth=0.5)
        ax.set_xlim3d(MINS[0], MAXS[0])
        ax.set_ylim3d(MINS[1], MAXS[1])
        ax.set_zlim3d(MINS[2], MAXS[2])
        """
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(-200, 200)
        """
        plt.title('{}-{:03d}'.format(title, i))
        cvs = plt2numpy(fig)
        plt.clf()
        cvss.append(cvs.astype("uint8"))
    if gif_path is not None:
        imageio.mimsave(gif_path, cvss, duration=duration)
    else:
        return cvss