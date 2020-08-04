import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class linear(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(linear, self).__init__()

        self.L1 = nn.Linear(n_inputs, n_outputs)
        init.xavier_uniform_(self.L1.weight)

    def forward(self, x):
        x = self.L1(x)
        return x


class mlp(nn.Module):
    def __init__(self, n_inputs, n_outputs, parameter_momentum=0.1):
        super(mlp, self).__init__()

        self.L1 = nn.Linear(n_inputs, 300, bias=False)
        init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(300, momentum=parameter_momentum)
        init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(300, 301, bias=False)
        init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(301, momentum=parameter_momentum)
        init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(301, 302, bias=False)
        init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(302, momentum=parameter_momentum)
        init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(302, 303, bias=False)
        init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(303, momentum=parameter_momentum)
        init.ones_(self.bn4.weight)

        self.L5 = nn.Linear(303, n_outputs, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)
        
    def forward(self, x):
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.L5(x)
        return x