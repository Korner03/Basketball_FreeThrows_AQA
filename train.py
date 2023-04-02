import argparse
import yaml
import numpy as np
import os
import os.path as osp
import random
import torch
from torch.utils.data import DataLoader
from utils.eval import eval_knn_accuracy, cycle
from utils.logging import log_accuracy, log_train_step
from moderator.moderator import UnifiedModerator
from model.model import create_student_unified_net, create_teacher_unified_net
from dataset.dataset import BBFTSDataset_Unified


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.yaml', help='path to config .yml file')

    args = parser.parse_args()
    return args


def init_framework(config):
    motion_teacher_net = create_teacher_unified_net(config=config)
    motion_student_net = create_student_unified_net(config=config)

    print('@ Loading Backbone Model Weights')
    model_CNN_pretrained_dict = torch.load(config['Model']['checkpoint_backbone'])
    model_CNN_dict = motion_teacher_net.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    motion_teacher_net.load_state_dict(model_CNN_dict)

    model_CNN_pretrained_dict = torch.load(config['Model']['checkpoint_backbone'])
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

    print('@ Loading Data 1...')
    train_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                                subset='train',
                                                                w_frozen=False),
                                   batch_size=config['Data']['train_set_len'],
                                   shuffle=False,
                                   num_workers=config['Data']['num_workers'],
                                   worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))
    data_train_full = next(cycle(train_loader_full))

    motion_feat_map_train = teacher_moder.val_func(data=data_train_full)[0][
        'motion_feat_map'].detach().cpu().numpy().reshape(config['Data']['train_set_len'], -1)
    motion_feat_map_train_dict = {}
    for i in range(config['Data']['train_set_len']):
        motion_feat_map_train_dict[int(data_train_full['name'][i])] = motion_feat_map_train[i, :]

    print('@ Loading Data 2...')
    val_loader_full = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                              subset='test',
                                                              w_frozen=False),
                                 batch_size=config['Data']['val_set_len'],
                                 shuffle=False,
                                 num_workers=config['Data']['num_workers'],
                                 worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))

    data_val_full = next(cycle(val_loader_full))
    motion_feat_map_val = teacher_moder.val_func(data=data_val_full)[0][
        'motion_feat_map'].detach().cpu().numpy().reshape(config['Data']['val_set_len'], -1)
    motion_feat_map_val_dict = {}
    for i in range(config['Data']['val_set_len']):
        motion_feat_map_val_dict[int(data_val_full['name'][i])] = motion_feat_map_val[i, :]

    print('@ Loading Data 3...')
    train_loader = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                           subset='train',
                                                           w_frozen=True,
                                                           motion_feat_map_dict=motion_feat_map_train_dict),
                              batch_size=config['Hyperparams']['batch_size'],
                              shuffle=True,
                              num_workers=config['Data']['num_workers'],
                              worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))

    print('@ Loading Data 4...')
    val_loader = DataLoader(dataset=BBFTSDataset_Unified(config=config,
                                                         subset='test',
                                                         w_frozen=True,
                                                         motion_feat_map_dict=motion_feat_map_val_dict),
                            batch_size=config['Data']['val_set_len'],
                            shuffle=False,
                            num_workers=config['Data']['num_workers'],
                            worker_init_fn=lambda _: np.random.seed(config['Hyperparams']['randomseed']))

    student_moder = UnifiedModerator(net=motion_student_net, lr=config['Hyperparams']['learning_rate'],
                                     a_student=config['Hyperparams']['loss_scale_student'],
                                     b_recons=config['Hyperparams']['loss_scale_recons'])

    return motion_student_net, train_loader_full, train_loader, val_loader_full, val_loader, student_moder


def train(config, model_student,
          train_loader_full, train_loader, val_loader_full,
          moder):
    print('============== Beginning Train ==============')

    example_ct = 0
    batch_ct = 0
    eval_frequency = 15
    for epoch in range(config['Hyperparams']['n_epochs']):
        for _, data in enumerate(train_loader):

            _, losses = moder.train_func(data)
            loss = sum(losses.values())
            loss_student = losses['student']
            loss_recons = losses['reconstruction']

            example_ct += len(data)
            batch_ct += 1

            if ((batch_ct + 1) % eval_frequency) == 0:
                correct, _, _ = eval_knn_accuracy(config=config, model_student=model_student,
                                                  val_loader_full=val_loader_full, train_loader_full=train_loader_full,
                                                  device=device)
                log_train_step(loss, loss_student, loss_recons, example_ct, epoch, correct)

    if not osp.exists(config['Experiments']['runs_path']):
        os.makedirs(config['Experiments']['runs_path'])

    outdir = osp.join(config['Experiments']['runs_path'],
                      str(len(os.listdir(config['Experiments']['runs_path'])) + 1))

    if not osp.exists(outdir):
        os.makedirs(outdir)

    output_model = osp.join(outdir, "model.pth")
    torch.save(model_student.state_dict(), output_model)
    print(f'@ Saved Model: {output_model}')


def init_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    init_random_seed(config['Hyperparams']['randomseed'])

    motion_student_net, train_loader_full, train_loader, val_loader_full, _, moder = init_framework(config=config)

    train(config, motion_student_net,
          train_loader_full, train_loader, val_loader_full,
          moder)

    correct, neg, pos = eval_knn_accuracy(config,
                                          motion_student_net, val_loader_full, train_loader_full,
                                          device=device)
    log_accuracy(correct=correct, neg=neg, pos=pos)
