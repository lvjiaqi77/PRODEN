import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Linearnet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Linearnet, self).__init__()

        self.L1 = nn.Linear(n_inputs, n_outputs)
        init.xavier_uniform_(self.L1.weight)


    def forward(self, x):
        x = self.L1(x)

        return x