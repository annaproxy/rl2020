import gym
import torch
from dqn import QNetwork
from torch import nn


class Trainer:

    def __init__(self, args):
        # Set gym environment TODO: infer input and output size from the env
        self.env = gym.envs.make(args.env)

        # Init  model
        self.model = QNetwork(args.input, args.output, args.hidden)

        self.loss = nn.CrossEntropyLoss()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)