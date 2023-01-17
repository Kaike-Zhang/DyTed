import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_feats, out_feats),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, pos, neg):
        pos = self.model(pos)
        neg = self.model(neg)

        return pos, neg
