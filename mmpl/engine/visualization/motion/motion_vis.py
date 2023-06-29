import glob
import os
import time
from typing import Any

import cv2
import mmcv
import mmengine
# os.environ['IMAGEIO_FFMPEG_EXE'] = '/Users/kyanchen/Documents/ffmpeg/ffmpeg'
# https://ffmpeg.org/download.html
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmpl.datasets.data_utils import lafan1_utils_torch
from mmpl.registry import HOOKS
from lightning.pytorch.callbacks import Callback
from .utilscv3d import draw_motion_based_global_pos


@HOOKS.register_module()
class MotionVisualizer(Callback):
    def __init__(self, save_dir, fps=30, *args, **kwargs):
        self.save_dir = save_dir
        mmengine.mkdir_or_exist(self.save_dir)
        self.fps = fps

    def fig2img(self, fig) -> np.ndarray:
        # convert matplotlib.figure.Figure to np.ndarray(cv2 format)
        fig.canvas.draw()
        graph_image = np.array(fig.canvas.get_renderer()._renderer)
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
        return graph_image

    def plot_pose(self, poses, prefix, parents):
        # z, y, x -> x, z, y
        # poses = np.transpose(poses, (2, 0, 1))
        np_imgs = []
        for i_frame, pose in enumerate(poses):
            pose = np.concatenate((poses[0], pose, poses[-1]), axis=0)

            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111, projection='3d')

            if parents is None:
                parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
            ax.cla()
            num_joint = pose.shape[0] // 3
            for i, p in enumerate(parents):
                if i > 0:
                    ax.plot([pose[i, 0], pose[p, 0]], \
                            [pose[i, 2], pose[p, 2]], \
                            [pose[i, 1], pose[p, 1]], c='r')
                    ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]], \
                            [pose[i + num_joint, 2], pose[p + num_joint, 2]], \
                            [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')
                    ax.plot([pose[i + num_joint * 2, 0], pose[p + num_joint * 2, 0]], \
                            [pose[i + num_joint * 2, 2], pose[p + num_joint * 2, 2]], \
                            [pose[i + num_joint * 2, 1], pose[p + num_joint * 2, 1]], c='g')
            ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1], c='b')
            ax.scatter(pose[num_joint:num_joint * 2, 0], pose[num_joint:num_joint * 2, 2],
                       pose[num_joint:num_joint * 2, 1], c='b')
            ax.scatter(pose[num_joint * 2:num_joint * 3, 0], pose[num_joint * 2:num_joint * 3, 2],
                       pose[num_joint * 2:num_joint * 3, 1], c='g')
            xmin = np.min(pose[:, 0])
            ymin = np.min(pose[:, 2])
            zmin = np.min(pose[:, 1])
            xmax = np.max(pose[:, 0])
            ymax = np.max(pose[:, 2])
            zmax = np.max(pose[:, 1])
            scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
            xmid = (xmax + xmin) // 2
            ymid = (ymax + ymin) // 2
            zmid = (zmax + zmin) // 2
            ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
            ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
            ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

            plt.draw()
            np_imgs.append(self.fig2img(fig))
            # plt.savefig(f"{prefix}_{i_frame:04}.png", dpi=200, bbox_inches='tight')
            plt.close()
        return np_imgs

    def save_gif(self, image_list, filepath, fps=30):
        imageio.mimsave(filepath, image_list, fps=fps)
        # # time.sleep(5)
        # frames = glob.glob(frames_prefix+'*.png')
        # frames.sort()
        # clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
        # clip.write_videofile(filepath)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pred_positions_shift, pred_rotations_shift, gt_positions_shift, gt_rotations_shift = outputs  # BxTx22x3, BxTx22x3x3, BxTx22x3, BxTx22x3x3
        parents = batch['parents'][0]
        bvh_files = batch['bvh_file']
        frame_idxs = batch['frame_idx']
        seq_idxs = batch['seq_idx']

        pred_g_rot, pred_g_pos = lafan1_utils_torch.fk_torch(pred_rotations_shift, pred_positions_shift, parents)
        gt_g_rot, gt_g_pos = lafan1_utils_torch.fk_torch(gt_rotations_shift, gt_positions_shift, parents)
        for idx in range(pred_g_pos.shape[0]):
            bvh_file = os.path.splitext(os.path.basename(bvh_files[idx]))[0]
            frame_idx = frame_idxs[idx].item()
            seq_idx = seq_idxs[idx].item()

            pred_pos = pred_g_pos[idx]
            pred_rot = pred_g_rot[idx]
            gt_pos = gt_g_pos[idx]
            gt_rot = gt_g_rot[idx]

            pred_gif = draw_motion_based_global_pos(
                pred_pos, parents,
                title=f"{bvh_file}_{frame_idx}_{seq_idx}_pred",
                axis_order=[0, 2, 1],
            )
            gt_gif = draw_motion_based_global_pos(
                gt_pos, parents,
                title=f"{bvh_file}_{frame_idx}_{seq_idx}_gt",
                axis_order=[0, 2, 1],
            )
            cvss = [np.hstack([pred_, gt_]) for pred_, gt_ in zip(pred_gif, gt_gif)]
            gif_path = f"{self.save_dir}/{bvh_file}_{frame_idx}_{seq_idx}_pred_gt.gif"
            duration = 1.0 / self.fps
            imageio.mimsave(gif_path, cvss, duration=duration)

