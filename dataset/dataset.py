import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentations import ToTensor, Resize, NormalizeMotion, RandomStaticPose


class BBFTSDataset_Unified(Dataset):
    def __init__(self, config, subset, w_frozen, motion_feat_map_dict={}):
        self.config = config
        self.motion_feat_map_dict = motion_feat_map_dict
        self.motions = {}
        self.motions_aligned = {}
        self.w_frozen = w_frozen
        self.frozen_aug_p = config['Hyperparams']['frozen_aug_p']

        nr_joints = config['Data']['pose']['nr_joints']
        data_dir = config['Data']['data_dir']

        motion_fpath = osp.join(data_dir, osp.join(subset, 'motion'))
        self.idx_to_names_map = []
        for curr_motion_name in os.listdir(motion_fpath):
            true_name = osp.splitext(curr_motion_name)[0]
            curr_motion = np.load(osp.join(motion_fpath, curr_motion_name))
            self.motions[true_name] = curr_motion
            self.idx_to_names_map.append(true_name)

        motion_fpath = osp.join(data_dir, osp.join(subset, 'aligned_motion'))
        for curr_motion_name in os.listdir(motion_fpath):
            true_name = osp.splitext(curr_motion_name)[0]
            self.motions_aligned[true_name] = np.load(osp.join(motion_fpath, curr_motion_name))

        self.n_samples = len(os.listdir(motion_fpath))

        df = pd.read_csv(osp.join(data_dir, config['Data']['labels_file']), header=0)

        self.labels_df = df.loc[df['phase'] == subset]

        self.pre_rel_n_frames = config['Hyperparams']['pre_release_n_frames']
        self.post_rel_n_frames = config['Hyperparams']['post_release_n_frames']

        mean_pose = np.load(osp.join(data_dir, config['Data']['meanpose_path']))
        std_pose = np.load(osp.join(data_dir, config['Data']['stdpose_path']))
        self.transforms_motion = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(nr_joints * 2, -1)),
            ToTensor()
        ])

        self.transforms_static = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(nr_joints * 2, -1)),
            ToTensor()
        ])

        self.transforms_frozen = transforms.Compose([
            RandomStaticPose(n_pose=1),
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(nr_joints * 2, -1)),
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
        aligned_motion = self.motions_aligned[str(vid_name)]

        orig_shot_frame = int(self.labels_df.loc[self.labels_df['video_name'] == int(vid_name)]['shot_frame'].item())

        if int(vid_name) == 1384:
            # TODO Permanently fix this sample or get rid of it
            last_motion_diff = aligned_motion[:, :, -1] - aligned_motion[:, :, -2]
            aligned_motion_ext = np.zeros(shape=(15, 2, 48))
            aligned_motion_ext[:, :, :44] = aligned_motion
            aligned_motion_ext[:, :, 44] = aligned_motion_ext[:, :, 43] + last_motion_diff
            aligned_motion_ext[:, :, 45] = aligned_motion_ext[:, :, 44] + last_motion_diff
            aligned_motion_ext[:, :, 46] = aligned_motion_ext[:, :, 45] + last_motion_diff
            aligned_motion_ext[:, :, 47] = aligned_motion_ext[:, :, 46] + last_motion_diff
            aligned_motion = aligned_motion_ext

        # extend motion by duplication (must have fixed number frames before shot release and after it)
        motion, shot_frame = self.duplicate_pose_by_shot_frame(motion, orig_shot_frame)
        aligned_motion, _ = self.duplicate_pose_by_shot_frame(aligned_motion, orig_shot_frame)

        # crop motion around shot release frame
        motion = motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]
        aligned_motion = aligned_motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]

        orig_motion = motion.copy()

        motion = self.trans_motion2d(motion2d=motion)
        motion = self.transforms_motion(motion)

        aligned_motion = self.trans_motion2d(motion2d=aligned_motion)

        if self.w_frozen and \
                np.random.choice([0, 1], 1, p=[1 - self.frozen_aug_p, self.frozen_aug_p]) == 1:
            aligned_motion = self.transforms_frozen(aligned_motion)
        else:
            aligned_motion = self.transforms_static(aligned_motion)

        sample = {'name': int(vid_name),
                  'motion': motion,
                  'aligned_motion': aligned_motion,
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