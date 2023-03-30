import argparse
import yaml
import numpy as np
import os.path as osp
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
from utils.eval import eval_knn_accuracy, cycle
from moderator.moderator import UnifiedModerator
from model.model import create_student_unified_net, create_teacher_unified_net
from dataset.dataset import BBFTSDataset_Unified


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.yml', help='path to config .yml file')

    args = parser.parse_args()
    return args


def init_framework(config):
    # TODO handle names of data dirs and magic numbers
    motion_teacher_net = create_teacher_unified_net(config=config)
    motion_student_net = create_student_unified_net(config=config)

    print('@ Loading Backbone weights')
    model_CNN_pretrained_dict = torch.load(config.checkpoint_backbone)
    model_CNN_dict = motion_teacher_net.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    motion_teacher_net.load_state_dict(model_CNN_dict)

    model_CNN_pretrained_dict = torch.load(config.checkpoint_backbone)
    model_CNN_dict = motion_student_net.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if
                                 k in model_CNN_dict and not k.startswith('mot_encoder')}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    motion_student_net.load_state_dict(model_CNN_dict)
    for name, param in motion_student_net.named_parameters():
        if name.startswith('static_encoder'):
            param.requires_grad = False
    print('@ Finished loading Weights')

    motion_teacher_net = motion_teacher_net.to(device)
    motion_student_net = motion_student_net.to(device)

    teacher_moder = UnifiedModerator(net=motion_teacher_net, lr=0, a_student=0, b_recons=0)
    print('Loading Data 1...')
    train_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                                subset='train_combined',
                                                                w_frozen=False),
                                   batch_size=config.train_set_len,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   worker_init_fn=lambda _: np.random.seed(config.randomseed))
    data_train_full = next(cycle(train_loader_full))

    motion_feat_map_train = teacher_moder.val_func(data=data_train_full)[0][
        'motion_feat_map'].detach().cpu().numpy().reshape(config.train_set_len, -1)
    motion_feat_map_train_dict = {}
    for i in range(config.train_set_len):
        motion_feat_map_train_dict[int(data_train_full['name'][i])] = motion_feat_map_train[i, :]
    print('Loading Data 2...')
    val_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                              subset='test_combined',
                                                              w_frozen=False),
                                 batch_size=config.val_set_len,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 worker_init_fn=lambda _: np.random.seed(config.randomseed))

    data_val_full = next(cycle(val_loader_full))
    motion_feat_map_val = teacher_moder.val_func(data=data_val_full)[0][
        'motion_feat_map'].detach().cpu().numpy().reshape(config.val_set_len, -1)
    motion_feat_map_val_dict = {}
    for i in range(config.val_set_len):
        motion_feat_map_val_dict[int(data_val_full['name'][i])] = motion_feat_map_val[i, :]
    print('Loading Data 3...')
    train_loader = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                           subset='train_combined',
                                                           w_frozen=True,
                                                           motion_feat_map_dict=motion_feat_map_train_dict),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              worker_init_fn=lambda _: np.random.seed(config.randomseed))
    print('Loading Data 4...')
    val_loader = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                         subset='test_combined',
                                                         w_frozen=True,
                                                         motion_feat_map_dict=motion_feat_map_val_dict),
                            batch_size=config.val_set_len,
                            shuffle=False,
                            num_workers=config.num_workers,
                            worker_init_fn=lambda _: np.random.seed(config.randomseed))

    student_moder = UnifiedModerator(net=motion_student_net, lr=config.learning_rate,
                                     a_student=config.loss_scale_student, b_recons=config.loss_scale_recons)

    return motion_student_net, train_loader_full, train_loader, val_loader_full, val_loader, student_moder


if __name__ == '__main__':
    args = parse_args()

    motion_student_net, train_loader_full, train_loader, val_loader_full, _, moder = init_framework(config=args.config)

    eval_knn_accuracy(args.config, motion_student_net, val_loader_full, train_loader_full, device=device, w_log=True)