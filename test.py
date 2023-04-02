import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.eval import eval_knn_accuracy, cycle
from utils.logging import log_accuracy
from moderator.moderator import UnifiedModerator
from model.model import create_student_unified_net, create_teacher_unified_net
from dataset.dataset import BBFTSDataset_Unified


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.yaml', help='path to config .yml file')
    parser.add_argument('--ckpt', type=str, default='Experiments/1/model.pth', help='path to checkpoint .pth file')

    args = parser.parse_args()
    return args


def init_test_framework(config, ckpt):
    motion_student_net = create_student_unified_net(config=config)

    print('@ Loading Model...')
    motion_student_net.load_state_dict(torch.load(ckpt))

    motion_student_net = motion_student_net.to(device)

    print('@ Loading Data...')
    train_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                                subset='train',
                                                                w_frozen=False),
                                   batch_size=config['Data']['train_set_len'],
                                   shuffle=False,
                                   num_workers=config['Data']['num_workers'],
                                   worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))

    val_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                              subset='test',
                                                              w_frozen=False),
                                 batch_size=config['Data']['val_set_len'],
                                 shuffle=False,
                                 num_workers=config['Data']['num_workers'],
                                 worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))

    return motion_student_net, train_loader_full, val_loader_full


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    motion_student_net, train_loader_full, val_loader_full = init_test_framework(config=config, ckpt=args.ckpt)

    correct, neg, pos = eval_knn_accuracy(config,
                                          motion_student_net, val_loader_full, train_loader_full,
                                          device=device)
    log_accuracy(correct=correct, neg=neg, pos=pos)
