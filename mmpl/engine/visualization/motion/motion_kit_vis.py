import glob
import os
import time
from typing import Any

import cv2
import mmcv
import mmengine
from mmpl.datasets.data_utils.kit.motion_process import recover_from_ric
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmpl.datasets.data_utils import lafan1_utils_torch
from mmpl.registry import HOOKS
from lightning.pytorch.callbacks import Callback
from .utilscv3d import draw_motion_based_global_pos


@HOOKS.register_module()
class MotionKITVisualizer(Callback):
    def __init__(
            self,
            save_dir,
            cache_dir=None,
            suffix='gif',
            fps=30,
            num_joints=21,
            *args, **kwargs):
        self.save_dir = save_dir
        self.suffix = suffix
        mmengine.mkdir_or_exist(self.save_dir)
        self.fps = fps
        self.num_joints = num_joints
        if num_joints == 21:
            self.parents = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
        elif num_joints == 22:
            self.parents = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        else:
            raise NotImplementedError
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            mmengine.mkdir_or_exist(self.cache_dir)

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
        pred_dict = {}
        for k, v in outputs.items():
            pred_dict[k] = v

        gt_motion = batch['motion']
        names = batch['name']
        frame_idxs = batch['frame_idx']
        seq_idxs = batch['seq_idx']

        pred_xyzs = {}
        for k, v in pred_dict.items():
            pred_xyzs[k] = recover_from_ric(v, self.num_joints).cpu().detach().numpy()

        gt_xyz = recover_from_ric(gt_motion, self.num_joints).cpu().detach().numpy()
        for idx in range(gt_xyz.shape[0]):
            name = names[idx]
            frame_idx = frame_idxs[idx].item()
            seq_idx = seq_idxs[idx].item()

            gt_pos = gt_xyz[idx]
            gt_gif = draw_motion_based_global_pos(
                gt_pos, self.parents,
                title=f"{name}_{frame_idx}_{seq_idx}_gt",
                axis_order=[0, 2, 1],
            )

            pred_gifs = []
            for k, pred_xyz in pred_xyzs.items():
                pred_pos = pred_xyz[idx]

                pred_gifs.append(draw_motion_based_global_pos(
                    pred_pos, self.parents,
                    title=f"{name}_{frame_idx}_{seq_idx}_{k}",
                    axis_order=[0, 2, 1])
                )
            max_len = max([len(pred_gif) for pred_gif in pred_gifs+[gt_gif]])
            cvss = []
            for i_cvs in range(max_len):
                cvs = []
                for pred_gif in pred_gifs:
                    if i_cvs < len(pred_gif):
                        cvs.append(pred_gif[i_cvs])
                    else:
                        cvs.append(pred_gif[-1])
                if i_cvs < len(gt_gif):
                    cvs.append(gt_gif[i_cvs])
                else:
                    cvs.append(gt_gif[-1])
                cvss.append(np.hstack(cvs))

            save_file_path = f"{self.save_dir}/{name}_{frame_idx}_{seq_idx}_pred_gt.{self.suffix}"
            duration = 1.0 / self.fps
            if self.suffix == "gif":
                imageio.mimsave(save_file_path, cvss, duration=duration)
            elif self.suffix == "mp4":
                writer = imageio.get_writer(save_file_path, fps=self.fps)
                for cvs in cvss:
                    writer.append_data(cvs)
                writer.close()

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pred_tokens = outputs.cpu().detach().numpy()
        assert len(pred_tokens) == 1, "batch size must be 1"
        name = batch['name'][0]
        np.save(os.path.join(self.cache_dir, name + '.npy'), pred_tokens)

