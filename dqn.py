import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        out = F.relu(out)
        out = self.l4(out)

        return out
