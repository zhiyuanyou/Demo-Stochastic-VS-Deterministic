import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class RobustLoss(nn.Module):
    def __init__(self, stochastic, num_max_sample, num_z_sample, epsilon):
        super().__init__()
        self.stochastic = stochastic
        self.num_max_sample = num_max_sample
        self.num_z_sample = num_z_sample
        self.epsilon = epsilon

    def cal_dis_z_delta(self, generator, real_x1, z, delta):
        gen_x2 = generator(real_x1, z)  # (n, 1)
        gen_x2_delta = generator(real_x1 + delta, z)  # (n, 1)
        dis = torch.norm(gen_x2 - gen_x2_delta, dim=1)  # (n, )
        return dis

    def cal_dis_avg_z(self, generator, real_x1, delta):
        if self.stochastic:
            real_x1 = torch.cat([real_x1] * self.num_z_sample, dim=0)  # (n, 1)
            z = Variable(Tensor(np.random.rand(real_x1.shape[0], 1)))  # (n, 1)
            dis = self.cal_dis_z_delta(generator, real_x1, z, delta)  # (n, )
            dis = torch.mean(dis, dim=0, keepdim=True)  # (1, )
        else:
            z = Variable(Tensor(np.zeros((real_x1.shape[0], 1))))  # (1, 1)
            dis = self.cal_dis_z_delta(generator, real_x1, z, delta)  # (1, )
        return dis

    def cal_dis_max_delta(self, generator, real_x1):
        dis_max = -float("inf")
        for delta in np.linspace(-self.epsilon, self.epsilon, self.num_max_sample):
            with torch.no_grad():
                dis = self.cal_dis_avg_z(generator, real_x1, delta)  # (1, )
                if dis > dis_max:
                    dis_max = dis
                    delta_max = delta
        return self.cal_dis_avg_z(generator, real_x1, delta_max)  # (1, )

    def forward(self, generator, real_x1_batch):
        dis_batch = []
        for real_x1 in real_x1_batch:
            dis = self.cal_dis_max_delta(generator, real_x1.unsqueeze(0))  # (1, )
            dis_batch.append(dis)
        dis_batch = torch.mean(torch.cat(dis_batch, dim=0))
        return dis_batch
