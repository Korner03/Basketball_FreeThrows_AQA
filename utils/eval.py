from sklearn.neighbors import NearestNeighbors
import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def check_knn_score(data_train, data_val, feat_map_train, feat_map_val, k):
    correct = 0
    pos = 0
    neg = 0

    neight = NearestNeighbors(n_neighbors=k).fit(feat_map_train)
    dists, out = neight.kneighbors(feat_map_val, return_distance=True)

    for i in range(out.shape[0]):
        curr_vote = data_train['cls_labels'].numpy()[out[i, :]].sum()
        curr_true_label = data_val['cls_labels'][i].item()
        if curr_vote > (k / 2) and curr_true_label == 1:
            correct += 1
            pos += 1
        elif curr_vote < (k / 2) and curr_true_label == 0:
            correct += 1
            neg += 1

    return correct, neg, pos


def eval_knn_accuracy(config, model_student, val_loader_full, train_loader_full, device):

    model_student.eval()
    with torch.no_grad():
        data_train_full = next(cycle(train_loader_full))
        motion_input = data_train_full['motion'].to(device)
        static_input = data_train_full['aligned_motion'].to(device)

        _, feat_map_train, _, _ = model_student(motion_input, static_input)
        feat_map_train = feat_map_train.detach().cpu().numpy().reshape(config['Data']['train_set_len'], -1)

        data_val_full = next(cycle(val_loader_full))
        motion_input = data_val_full['motion'].to(device)
        static_input = data_val_full['aligned_motion'].to(device)

        _, feat_map_val, _, _ = model_student(motion_input, static_input)
        feat_map_val = feat_map_val.detach().cpu().numpy().reshape(config['Data']['val_set_len'], -1)

        correct, neg, pos = check_knn_score(data_train=data_train_full, data_val=data_val_full,
                                            feat_map_train=feat_map_train, feat_map_val=feat_map_val,
                                            k=config['Hyperparams']['knn_k'])

    return correct, neg, pos
