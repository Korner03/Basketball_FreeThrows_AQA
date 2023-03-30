import numpy as np
import torch


class NormalizeMotion(object):
    """
    Normalizes motion sample using Mean and Standard Deviation
    """
    def __init__(self, mean_pose, std_pose):
        """
        :param mean_pose: (J, 2)
        :param std_pose: (J, 2)
        """
        self.mean_pose = mean_pose
        self.std_pose = std_pose

    def __call__(self, motion):
        """
        :param motion: (J, 2, T)
        :param mean_pose: (J, 2)
        :param std_pose: (J, 2)
        :return: Normalized motion (J, 2, T)
        """
        return (motion - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Mean: {self.mean_pose},'
            f'Variance: {self.std_pose})'
            )
        return repr_str


class Resize(object):
    """
    Resizes given sample
    """

    def __init__(self, scale):
        """
        :param scale: tuple
        """
        self.scale = scale

    def __call__(self, sample):
        """
        Default value for resize is (Joints * 2, T)
        :param sample: input sample (np.array)
        :return: resized sample
        """
        return sample.reshape(self.scale)

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Scale: {self.scale})'
            )
        return repr_str


class ToTensor(object):
    """
    Converts ndarrays in sample to Tensor
    """

    def __call__(self, sample):
        """
        Default value for resize is (Joints * 2, T)
        :param sample: input sample (np.array)
        :return: sample as 3D Tensor
        """
        return torch.Tensor(sample)


class RandomStaticPose(object):
    """
    Applies random Rotation
    """

    def __init__(self, n_pose):
        """
        """
        self.n_pose = n_pose

    def __call__(self, sample):
        """
        :param sample: input sample (np.array)
        :return: masked sample
        """
        _, _, n_frames = sample.shape
        pose_idx = np.random.choice(np.arange(n_frames), 1)
        selected_pose = sample[:, :, pose_idx]
        sample[:, :, :] = selected_pose

        return sample

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'n_pose: {self.n_pose},'
            f'n_pose: {self.n_pose})'
        )
        return repr_str