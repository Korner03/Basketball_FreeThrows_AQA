import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, channels, dropout_p, kernel_size=7):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=dropout_p))
            if not i == len(channels) - 2:
                model.append(acti)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None):
        super(Encoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 1
        stride = 2

        for i in range(nr_layer):
            if convpool is None:
                pad = (kernel_size - 2) // 2
                if pad == 0 or pad == 1:
                    pad += 1
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1],
                                       kernel_size=kernel_size, stride=1))
                model.append(acti)
                model.append(convpool(kernel_size=2, stride=2))

        self.global_pool = global_pool

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)

        return x


class UnifiedNetwork(nn.Module):
    def __init__(self, mot_en_channels, stat_en_channels, de_channels, mot_kernel_size,
                 do_p, global_pool=None, convpool=None):
        super(UnifiedNetwork, self).__init__()

        self.mot_encoder = Encoder(channels=mot_en_channels, kernel_size=mot_kernel_size)
        self.static_encoder = Encoder(channels=stat_en_channels, kernel_size=7,
                                      global_pool=global_pool, convpool=convpool)
        self.decoder = Decoder(channels=de_channels, dropout_p=do_p)

    def forward(self, x_motion, x_static):
        mot_feat_map = self.mot_encoder(x_motion)

        static_feat_map = self.static_encoder(x_static[:, :-2, :])
        static_feat_map_dupl = static_feat_map.repeat(1, 1, mot_feat_map.shape[-1])

        cat_feat_map = torch.cat([mot_feat_map, static_feat_map_dupl], dim=1)

        decoder_output = self.decoder(cat_feat_map)

        return decoder_output, cat_feat_map, \
               mot_feat_map.reshape(mot_feat_map.shape[0], -1), static_feat_map.reshape(static_feat_map.shape[0], -1)


def create_teacher_unified_net(config):
    return UnifiedNetwork(mot_en_channels=config['Model']['mot_en_channels'],
                          stat_en_channels=config['Model']['stat_en_channels'],
                          de_channels=config['Model']['de_channels'],
                          mot_kernel_size=8, do_p=config['Hyperparams']['do_p'],
                          global_pool=F.max_pool1d, convpool=nn.MaxPool1d)


def create_student_unified_net(config):
    return UnifiedNetwork(mot_en_channels=config['Model']['mot_en_channels'],
                          stat_en_channels=config['Model']['stat_en_channels'],
                          de_channels=config['Model']['de_channels'],
                          mot_kernel_size=3, do_p=config['Hyperparams']['do_p'],
                          global_pool=F.max_pool1d, convpool=nn.MaxPool1d)