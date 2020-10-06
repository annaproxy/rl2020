from argparse import ArgumentParser
import numpy as np
import torch
import random
from Trainer import Trainer


def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    trainer = Trainer(args)

    duration = trainer.train(100)
    print(duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--input', type=int, required=True,
    #                   help='Specify input size')
    # parser.add_argument('--output', type=int, required=True,
    #                   help='Specify output size')
    parser.add_argument('--hidden', type=int, default=128,
                      help='Specify hidden size')
    parser.add_argument('--memory', type=int, default=10000,
                      help='Specify memory size for memory replay')
    parser.add_argument('--env', type=str, choices=['MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1'],
                        default='MountainCar-v0', help='Specify gym environment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learn rate')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='epsilon for eps-greedy')
    parser.add_argument('--discount', type=float, default=0.8,
                        help='discount factor')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    args = parser.parse_args()
    main(args)
