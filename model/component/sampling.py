import torch
from torch import nn
from torch.nn import Parameter

import numpy as np


class Sampling(nn.Module):

    def __init__(self, args):
        super(Sampling, self).__init__()
        self.a = Parameter(torch.FloatTensor([3/8]).to(args.device), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
        self.t = 0.01

    def forward(self, total_len):
        seq_begin = np.random.randint(0, total_len)
        seq_end = np.random.randint(0, total_len)
        while np.abs(seq_begin - seq_end + 1) < 4:
            seq_end = np.random.randint(0, total_len)
        if seq_begin > seq_end:
            temp = seq_begin
            seq_begin = seq_end
            seq_end = temp
        seq_len = seq_end - seq_begin + 1
        p = self.a * (seq_len-2)/seq_len + 3/4 - 1/3 * self.a
        # p.clip_(0.0, 0.99)
        smp_list_1 = (torch.hstack([torch.pow(p, i) * (1 - p) for i in range(seq_len)]) /
                      (1 - torch.pow(p, seq_len))).to(self.a.device)
        smp_list_2 = (torch.hstack([torch.pow(p, seq_len-i-1) * (1 - p) for i in range(seq_len)]) /
                      (1 - torch.pow(p, seq_len))).to(self.a.device)

        epsilon_1 = torch.log(-1 * torch.log(torch.rand(seq_len))).to(self.a.device)
        epsilon_2 = torch.log(-1 * torch.log(torch.rand(seq_len))).to(self.a.device)

        len_1 = self.softmax((smp_list_1 - epsilon_1) / self.t)
        seq_1_end = torch.argmax(len_1)
        padding_1 = torch.zeros_like(len_1).to(self.a.device)
        padding_1[0:seq_1_end] = 1
        len_1 = len_1 + padding_1

        len_2 = torch.flip(self.softmax((smp_list_2 - epsilon_2) / self.t), dims=[0])
        seq_2_begin = torch.argmax(len_2)
        padding_2 = torch.zeros_like(len_2).to(self.a.device)
        padding_2[seq_2_begin+1:] = 1
        len_2 = len_2 + padding_2

        if len(len_1[len_1.isnan()]) > 0:
            len_1 = torch.ones_like(len_1)
        if len(len_2[len_2.isnan()]) > 0:
            len_2 = torch.ones_like(len_2)

        return seq_begin, seq_1_end, seq_2_begin, seq_end, len_1, len_2

    def forward_rd(self, total_len):
        seq_begin = np.random.randint(0, total_len)
        seq_end = np.random.randint(0, total_len)
        while np.abs(seq_begin - seq_end + 1) < 4:
            seq_end = np.random.randint(0, total_len)
        if seq_begin > seq_end:
            temp = seq_begin
            seq_begin = seq_end
            seq_end = temp
        k = seq_end - seq_begin + 1
        seq_1_end = np.random.randint(1, k)
        seq_2_begin = np.random.randint(1, k)

        len_1 = torch.zeros(seq_end - seq_begin + 1)
        len_2 = torch.zeros(seq_end - seq_begin + 1)
        len_1[:seq_1_end+1] = 1
        len_2[seq_2_begin:] = 1

        return seq_begin, seq_1_end, seq_2_begin, seq_end, len_1, len_2

    def forward_bo(self, total_len):
        seq_begin = np.random.randint(0, total_len)
        seq_end = np.random.randint(0, total_len)
        while np.abs(seq_begin - seq_end + 1) < 4:
            seq_end = np.random.randint(0, total_len)
        if seq_begin > seq_end:
            temp = seq_begin
            seq_begin = seq_end
            seq_end = temp
        seq_len = seq_end - seq_begin + 1
        p = torch.tensor(0.75)
        # p.clip_(0.0, 0.99)
        smp_list_1 = (torch.hstack([torch.pow(p, i) * (1 - p) for i in range(seq_len)]) /
                      (1 - torch.pow(p, seq_len))).to(self.a.device)
        smp_list_2 = (torch.hstack([torch.pow(p, seq_len - i - 1) * (1 - p) for i in range(seq_len)]) /
                      (1 - torch.pow(p, seq_len))).to(self.a.device)

        epsilon_1 = torch.log(-1 * torch.log(torch.rand(seq_len))).to(self.a.device)
        epsilon_2 = torch.log(-1 * torch.log(torch.rand(seq_len))).to(self.a.device)

        len_1 = self.softmax((smp_list_1 - epsilon_1) / self.t)
        seq_1_end = torch.argmax(len_1)
        padding_1 = torch.zeros_like(len_1).to(self.a.device)
        padding_1[0:seq_1_end] = 1
        len_1 = len_1 + padding_1

        len_2 = torch.flip(self.softmax((smp_list_2 - epsilon_2) / self.t), dims=[0])
        seq_2_begin = torch.argmax(len_2)
        padding_2 = torch.zeros_like(len_2).to(self.a.device)
        padding_2[seq_2_begin + 1:] = 1
        len_2 = len_2 + padding_2

        if len(len_1[len_1.isnan()]) > 0:
            len_1 = torch.ones_like(len_1)
        if len(len_2[len_2.isnan()]) > 0:
            len_2 = torch.ones_like(len_2)

        return seq_begin, seq_1_end, seq_2_begin, seq_end, len_1, len_2







