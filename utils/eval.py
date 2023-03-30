import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def check_knn_score(data_train, data_val, feat_map_train, feat_map_val,
                    k, val_names, print_names, w_dists=False):
    # TODO sort out strings below and useless variables?
    correct = 0
    pos = 0
    neg = 0

    neight = NearestNeighbors(n_neighbors=k).fit(feat_map_train)
    dists, out = neight.kneighbors(feat_map_val, return_distance=True)

    for i in range(out.shape[0]):
        curr_vote = data_train['cls_labels'].numpy()[out[i, :]].sum()
        curr_true_label = data_val['cls_labels'][i].item()
        curr_train_names = data_train['name'][out[i, :]]
        if curr_vote > (k / 2) and curr_true_label == 1:
            correct += 1
            pos += 1
        elif curr_vote < (k / 2) and curr_true_label == 0:
            correct += 1
            neg += 1
        curr_val_name = data_val['name'][i]
        if print_names and curr_val_name in val_names:
            print(f'{curr_val_name} (KNN Vote: {curr_vote}, GT: {curr_true_label}), Names: {curr_train_names}')
            if w_dists:
                np.set_printoptions(precision=3)
                print(dists[i, :])

    return correct, neg, pos


def eval_knn_accuracy(config, model_student, val_loader_full, train_loader_full, device, w_log):
    # TODO fix this - select K upfront
    if w_log:
        print('============== Performing Test ==============')
    corrects_list = []

    model_student.eval()
    with torch.no_grad():
        data_train_full = next(cycle(train_loader_full))
        motion_input = data_train_full['motion'].to(device)
        static_input = data_train_full['smoothed_motion'].to(device)

        _, feat_map_train, _, _ = model_student(motion_input, static_input)
        feat_map_train = feat_map_train.detach().cpu().numpy().reshape(config.train_set_len, -1)

        data_val_full = next(cycle(val_loader_full))
        motion_input = data_val_full['motion'].to(device)
        static_input = data_val_full['smoothed_motion'].to(device)

        _, feat_map_val, _, _ = model_student(motion_input, static_input)
        feat_map_val = feat_map_val.detach().cpu().numpy().reshape(config.val_set_len, -1)

        for curr_k in [3, 5, 7, 9]:
            correct, neg, pos = check_knn_score(data_train=data_train_full, data_val=data_val_full,
                                                feat_map_train=feat_map_train, feat_map_val=feat_map_val,
                                                k=curr_k, val_names=[], print_names=False)
            corrects_list.append(correct)

        print(f'\nValidation Results: {corrects_list} - Mean: {sum(corrects_list) / len(corrects_list)}')

        if w_log:
            wandb.log({'Final Accuracy': {'K=3 Accuracy': corrects_list[0],
                                          'K=5 Accuracy': corrects_list[1],
                                          'K=7 Accuracy': corrects_list[2],
                                          'K=9 Accuracy': corrects_list[3],
                                          'MeanKAcc': sum(corrects_list) / len(corrects_list)}})
        # TODO: here i can save if i need to..
    return corrects_list