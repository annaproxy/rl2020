from argparse import ArgumentParser
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
import json

from Trainer import Trainer
from utils import smooth

def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initiate NO REPLAY mode 
    if args.no_replay:
        args.batch_size = 1
        args.memory = 1

    # Print the setup for the record
    print('-----------------------\nUsed hyperparameter setup:')
    print(f'{vars(args)}\n-----------------------')

    args.experiment_directory = f'experiments/{json.dumps(vars(args))[1:-1]}'
    args.experiment_directory = args.experiment_directory.replace('"','')
    args.experiment_directory = args.experiment_directory.replace(': ','_')
    os.makedirs(args.experiment_directory,
                exist_ok=True)
    os.makedirs(args.experiment_directory+'/models/',
                exist_ok=True)

    trainer = Trainer(args)

    duration = trainer.train(args.episodes)
    print(duration)

    # save results
    np.save(f'{args.experiment_directory}/ep_durations', duration)

    # plot results
    plt.figure(figsize=(20, 10))
    plt.plot(smooth(duration, 10))
    plt.ylabel('Steps')
    plt.xlabel('Episodes')
    plt.title(f'{args.env}')
    plt.savefig(f'{args.experiment_directory}/results.png')
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hidden', type=int, default=128,
                      help='Specify hidden size')
    parser.add_argument('--memory', type=int, default=10000,
                      help='Specify memory size for experience replay')
    parser.add_argument('--env', type=str, choices=['MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1','Acrobot-v1'],
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
    parser.add_argument('--episodes', type=int, default=100,
                        help='batch size')
    parser.add_argument('--epsilon_cap', type=int, default=1000,
                        help='threhold for epsilon reduction')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='maximum steps per episode')
    parser.add_argument('--target', action='store_true',
                        help='activate target network')
    parser.add_argument('--no_replay', action='store_true',
                        help='dont activate replay')
    parser.add_argument('--C', type=int, default=10,
                        help='how many times to save the target network')
    parser.add_argument('--save_amt', type=int, default=10,
                        help='save the weights to file')
                        
    args = parser.parse_args()
    main(args)
