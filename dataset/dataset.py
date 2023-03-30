import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from augmentations import ToTensor, Resize, NormalizeMotion, RandomStaticPose


class BBFTSDataset_Unified(Dataset):
    def __init__(self, config, subset, w_frozen, motion_feat_map_dict={}):
        # TODO caller should send 'config'
        self.config = config
        self.motion_feat_map_dict = motion_feat_map_dict
        self.motions = {}
        self.motions_smoothed = {}
        self.w_frozen = w_frozen

        motion_dir = 'motion'  # TODO
        motion_fpath = osp.join(config.data_dir, osp.join(subset, motion_dir))
        self.idx_to_names_map = []
        for curr_motion_name in os.listdir(motion_fpath):
            true_name = osp.splitext(curr_motion_name)[0]
            curr_motion = np.load(osp.join(motion_fpath, curr_motion_name))
            self.motions[true_name] = curr_motion
            self.idx_to_names_map.append(true_name)

        motion_dir = f'smoothed_motion_v8'  # TODO

        motion_fpath = osp.join(config.data_dir, osp.join(subset, motion_dir))
        for curr_motion_name in os.listdir(motion_fpath):
            true_name = osp.splitext(curr_motion_name)[0]
            self.motions_smoothed[true_name] = np.load(osp.join(motion_fpath, curr_motion_name))

        self.n_samples = len(os.listdir(motion_fpath))

        df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)

        self.labels_df = df.loc[df['phase'] == subset]
        self.labels_train_df = df.loc[df['phase'] == 'train_combined']

        self.pre_rel_n_frames = config.pre_release_n_frames
        self.post_rel_n_frames = config.post_release_n_frames + 3

        mean_pose = np.load(config.meanpose_path)
        std_pose = np.load(config.stdpose_path)
        self.transforms_motion = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            ToTensor()
        ])

        self.transforms_static = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            ToTensor()
        ])

        self.transforms_frozen = transforms.Compose([
            RandomStaticPose(n_pose=1),
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            ToTensor()
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        vid_name = self.idx_to_names_map[idx]
        motion_feat_map = torch.empty(size=(3, 3))
        if len(self.motion_feat_map_dict.keys()) > 0 and int(vid_name) in self.motion_feat_map_dict.keys():
            motion_feat_map = self.motion_feat_map_dict[int(vid_name)]
            motion_feat_map = torch.from_numpy(motion_feat_map)

        # Create one-hot label vector
        label_idx = int(self.labels_df.loc[self.labels_df['video_name'] == int(vid_name)]['label'].item())
        cls_label = torch.from_numpy(np.array([label_idx])).type(torch.long)

        motion = self.motions[str(vid_name)]
        smoothed_motion = self.motions_smoothed[str(vid_name)]

        orig_shot_frame = int(self.labels_df.loc[self.labels_df['video_name'] == int(vid_name)]['shot_frame'].item())

        if int(vid_name) == 1384:
            # TODO
            last_motion_diff = smoothed_motion[:, :, -1] - smoothed_motion[:, :, -2]
            smoothed_motion_ext = np.zeros(shape=(15, 2, 48))
            smoothed_motion_ext[:, :, :44] = smoothed_motion
            smoothed_motion_ext[:, :, 44] = smoothed_motion_ext[:, :, 43] + last_motion_diff
            smoothed_motion_ext[:, :, 45] = smoothed_motion_ext[:, :, 44] + last_motion_diff
            smoothed_motion_ext[:, :, 46] = smoothed_motion_ext[:, :, 45] + last_motion_diff
            smoothed_motion_ext[:, :, 47] = smoothed_motion_ext[:, :, 46] + last_motion_diff
            smoothed_motion = smoothed_motion_ext

        # extend motion by duplication (must have fixed number frames before shot release and after it)
        motion, shot_frame = self.duplicate_pose_by_shot_frame(motion, orig_shot_frame)
        smoothed_motion, _ = self.duplicate_pose_by_shot_frame(smoothed_motion, orig_shot_frame)

        # crop motion around shot release frame
        motion = motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]
        smoothed_motion = smoothed_motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]

        orig_motion = motion.copy()

        motion = self.trans_motion2d(motion2d=motion)
        motion = self.transforms_motion(motion)

        smoothed_motion = self.trans_motion2d(motion2d=smoothed_motion)

        if self.w_frozen and \
                np.random.choice([0, 1], 1, p=[1 - self.config.frozen_aug_p, self.config.frozen_aug_p]) == 1:
            smoothed_motion = self.transforms_frozen(smoothed_motion)
        else:
            smoothed_motion = self.transforms_static(smoothed_motion)

        sample = {'name': int(vid_name),
                  'motion': motion,
                  'smoothed_motion': smoothed_motion,
                  'orig_motion': orig_motion,
                  'teacher_motion_ft_map': motion_feat_map,
                  'cls_labels': cls_label
                  }

        return sample

    def trans_motion2d(self, motion2d):
        # subtract centers to local coordinates
        centers = motion2d[8, :, :]
        motion_proj = motion2d - centers

        # adding velocity
        velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
        motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

        return motion_proj

    def duplicate_pose_by_shot_frame(self, motion, shot_frame):
        if shot_frame <= self.pre_rel_n_frames:
            n_diff = self.pre_rel_n_frames - shot_frame
            shot_frame += n_diff
            first_pose = np.copy(motion[:, :, 0])
            first_pose = first_pose[..., np.newaxis]
            first_pose_matrix = np.repeat(first_pose, n_diff, axis=2)
            motion = np.concatenate((first_pose_matrix, motion), axis=2)

        return motion, shot_frame