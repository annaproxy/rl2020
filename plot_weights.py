import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import sys

dirname_noreplay = """"hidden": 32, "memory": 1, "env": "CartPole-v1", "seed": 42, "lr": 0.001, "epsilon": 0.5,\
 "discount": 0.8, "batch_size": 1, "episodes": 6000, "epsilon_cap": 1000, "max_steps": 250, "target": false,\
 "no_replay": true, "C": 10, "save_amt": 10"""
dirname_replay = """"hidden": 32, "memory": 10000, "env": "CartPole-v1", "seed": 42, "lr": 0.001, "epsilon": 0.5,\
 "discount": 0.8, "batch_size": 64, "episodes": 6000, "epsilon_cap": 1000, "max_steps": 250, "target": false,\
 "no_replay": false, "C": 10, "save_amt": 10"""


def plot_two(
    dirname,
    hidden_size,
    save_amount,
    episode_amount,
    name="replay",
    l1_index=0,
    l2_index=0,
):
    """
    Plots the weights from the two layers.


    dirname : directoryname for the experiment
    hidden_size : Size of hidden layer
    save_amount : Save_amt, how many models were saved
    episode_amount : how many episodes are available
    name : For pdf description
    l1_index, l2_index: Depends on your problem representation. for cartpole, l1.weight has shape hidden_size x 4
                        so l1_index can be 0,1,2,3
    """
    l1_matrix = np.zeros((hidden_size, 1))
    l2_matrix = np.zeros((hidden_size, 1))
    for i in range(0, episode_amount, save_amount):
        hi = torch.load("experiments/" + dirname + f"/models/episode{i}.pt")
        l1_weights = np.expand_dims(hi["l1.weight"].cpu().numpy()[:, l1_index], 1)
        l2_weights = np.expand_dims(hi["l2.weight"].cpu().numpy()[l2_index, :], 1)
        l1_matrix = np.concatenate((l1_matrix, l1_weights), axis=1)
        l2_matrix = np.concatenate((l2_matrix, l2_weights), axis=1)

    for i in range(hidden_size):
        current = l1_matrix[i]
        current2 = l2_matrix[i]
        plt.plot(current, label=f"l1_w{i}")
        plt.plot(current2, label=f"l2_w{i}")

    plt.savefig(f"weights_{name}_{l1_index}_{l2_index}.pdf")


#plot_two(dirname_replay, 32, 10, 6000)
#plot_two(dirname_noreplay, 32, 10, 6000, name="noreplay")
