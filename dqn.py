import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers=0):
        nn.Module.__init__(self)
        modules = []
        # self.l1 = nn.Linear(input_size, hidden_size)
        # self.l2 = nn.Linear(hidden_size, output_size)
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(nn.ReLU())
        for i in range(num_layers):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        # out = self.l1(x)
        # out = F.relu(out)
        # out = self.l2(out)
        out = self.network(x)

        return out

    #
    # def __init__(self, input_size, hidden_layers, output_size=1):
    #     super(LTRModel, self).__init__()
    #
    #     self.input_size = input_size
    #     self.hidden_layers = hidden_layers
    #     self.output_size = output_size
    #
    #     layers = [self.input_size] + self.hidden_layers + [self.output_size]
    #     modules = list()
    #     for i in range(1, len(layers) - 1):
    #         modules.append(nn.Linear(layers[i-1], layers[i]))
    #         modules.append(nn.ReLU())
    #     modules.append(nn.Linear(layers[-2], layers[-1]))
    #     self.network = nn.Sequential(*modules)
    #
    # def forward(self, x):
    #     return self.network(x)