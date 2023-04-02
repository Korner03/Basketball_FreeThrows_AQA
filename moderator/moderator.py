import torch
import torch.nn as nn
import torch.optim as optim


class UnifiedModerator(object):
    def __init__(self, net, lr, a_student, b_recons):
        self.net = net

        self.loss_recons = nn.L1Loss()
        self.loss_student = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr)

        self.a_student = a_student
        self.b_recons = b_recons
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        motion_input = data['motion'].to(self.device)
        static_input = data['aligned_motion'].to(self.device)
        cls_label = data['cls_labels'].to(self.device).reshape(-1)
        teacher_feat_map = data['teacher_motion_ft_map'].to(self.device)

        output_recons, cat_feat_map, motion_feat_map, static_feat_map = self.net(motion_input, static_input)

        losses = None
        if self.a_student != 0 and len(teacher_feat_map.shape) == 2:
            loss_student = self.loss_student(motion_feat_map, teacher_feat_map) * self.a_student
            loss_recon = self.loss_recons(output_recons, motion_input) * self.b_recons

            losses = {'reconstruction': loss_recon,
                      'student': loss_student
                      }

        outputs = {'reconstruction': output_recons,
                   'cat_feat_map': cat_feat_map,
                   'motion_feat_map': motion_feat_map,
                   'static_feat_map': static_feat_map,
                   'cls_label': cls_label
                   }

        return outputs, losses

    def val_func(self, data):
        self.net.eval()

        with torch.no_grad():
            output, losses = self.forward(data)

        return output, losses

    def train_func(self, data):
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)

        return outputs, losses

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()