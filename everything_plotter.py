import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import sys
import random
from matplotlib.colors import ListedColormap
import seaborn as sns
from utils import smooth


class EverythingPlotter:
    def __init__(
        self,
        hidden=128,
        lr=0.001,
        env="CartPole-v1",
        discount=0.8,
        episodes=10000,
        max_steps=200,
        C=10,
        save_amt=10,
        state_size=4,
        action_size=2,
        seed_list=[2, 12, 22, 32, 42],
        experiment_dir="experiments",
    ):
        """
        Given all the relevant parameters (that are NOT varied for our experiments, i.e. hidden size, learning rate, env),
        constructs an EverythingPlotter.
        You also need the state_size and action_size for plotting the weights.
        Currently, plotting weights only works for a two layer network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = episodes
        self.hidden = hidden
        self.save_amt = save_amt
        self.seed_list = seed_list
        self.max_steps = max_steps
        self.experiment_dir = experiment_dir
        self.dir_name = (
            "hidden_"
            + str(hidden)
            + ", memory_{memory}, env_"
            + str(env)
            + ", seed_{seed}, lr_"
            + str(lr)
            + ", epsilon_0.5, discount_"
            + str(discount)
            + ", batch_size_{batch_size}, episodes_"
            + str(episodes)
            + ", epsilon_cap_1000, max_steps_"
            + str(max_steps)
            + ", target_{target}, no_replay_{no_replay}, C_"
            + str(C)
            + ", save_amt_"
            + str(save_amt)
        )
        print("EverythingPlotter loaded")

    def _get_weights_dict(self, seed):
        """
        Get a dictionary of (memory, batch_size, target) to a matrix of weights during training.
        Inputs: seed = the seed for which to extract the weights, as averaging is difficult over randomly initialized weights.
        Outputs:
        """
        weights_dict = dict()
        amount_saved = int(self.episodes / self.save_amt)
        for memory in [1, 10000]:
            for batch_size in [1, 64]:
                for target in ["true", "false"]:
                    if memory == 1 and batch_size == 64:
                        continue
                    if memory == 1 and batch_size == 1:
                        no_replay = "true"
                    else:
                        no_replay = "false"
                    filename = self.dir_name.format(
                        memory=memory,
                        batch_size=batch_size,
                        seed=seed,
                        no_replay=no_replay,
                        target=target,
                    )
                    episodes = np.zeros(
                        (
                            amount_saved,
                            (self.hidden * self.state_size)
                            + (self.hidden * self.action_size),
                        )
                    )
                    for i in range(0, amount_saved, self.save_amt):
                        final_filename = (
                            f"{self.experiment_dir}/{filename}/models/episode{i}.pt"
                        )
                        if not os.path.exists(final_filename):
                            warnings.warn(
                                "Skipping batch_size=1, memory=10k experiment"
                            )
                            continue
                        hi = torch.load(final_filename)
                        w1 = hi["l1.weight"].cpu().numpy()
                        w2 = hi["l2.weight"].cpu().numpy()
                        flattened_w1 = w1.flatten()
                        flattened_w2 = w2.flatten()
                        all_weights = np.concatenate((flattened_w1, flattened_w2))
                        episodes[i] = all_weights
                    weights_dict[(memory, batch_size, target)] = episodes
        return weights_dict

    def get_descriptive_tuple(self, tuple):
        # (memory, batch_size, target)
        part1 = (
            "No replay"
            if tuple[0] == 1 and tuple[0] == 1
            else f"Replay, batch_size={tuple[1]}"
        )
        return part1 + (
            " with target network" if tuple[2] == "true" else " without target network"
        )

    def _get_scores_dict(self):
        scores_dict = dict()
        for memory in [1, 10000]:
            for batch_size in [1, 64]:
                for target in ["true", "false"]:
                    if memory == 1 and batch_size == 64:
                        continue
                    if memory == 1 and batch_size == 1:
                        no_replay = "true"
                    else:
                        no_replay = "false"
                    episodes = np.zeros((len(self.seed_list), self.episodes))
                    for i, seed in enumerate(self.seed_list):
                        filename = self.dir_name.format(
                            memory=memory,
                            batch_size=batch_size,
                            seed=seed,
                            no_replay=no_replay,
                            target=target,
                        )
                        final_filename = (
                            f"{self.experiment_dir}/{filename}/ep_durations.npy"
                        )
                        if not os.path.exists(final_filename):
                            warnings.warn(
                                "Skipping batch_size=1, memory=10k experiment"
                            )
                            continue
                        durations = np.load(final_filename)

                        episodes[i] = durations
                    scores_dict[(memory, batch_size, target)] = episodes
        return scores_dict

    def plot_weights(self, smoothing=200, filename="THICK"):
        """
        Plots all weights for all seeds, in separate files
        """
        sns.set_style("darkgrid")
        for seed in self.seed_list:
            fig, axs = plt.subplots(3, 2, figsize=(5 * 2, 5 * 3))
            weights_dict = self._get_weights_dict(seed)
            for ax, z in zip(axs.flat, weights_dict):
                for w in weights_dict[z].T:
                    w_s = smooth(w, smoothing)
                    ax.plot(np.arange(0, len(w_s)), w_s)
                    ax.set_title(self.get_descriptive_tuple(z))

            plt.savefig(f"{filename}_seed{seed}.pdf")

    def plot_replay_variations(self, smoothing=500, filename="cartpole"):
        """
        Plots all replay variations (when applicable)
        """
        scores_dict = self._get_scores_dict()

        sns.set_style("darkgrid")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5, 3))
        colors = sns.color_palette("Set1", 6)
        colors = list(colors.as_hex())
        for i, z in enumerate(scores_dict):
            if z[0] == 1 and z[1] == 1:
                ax = ax1
            elif z[1] == 1:
                ax = ax2
            else:
                ax = ax3
            the_means = np.mean(scores_dict[z], axis=0)
            the_sds = np.std(scores_dict[z], axis=0)
            ax.plot(
                smooth(the_means, smoothing),
                label="With target net" if z[2] == "true" else "Without target net ",
                color=colors[i],
            )
            ax.fill_between(
                np.arange(len(smooth(the_means, smoothing))),
                smooth(the_means - the_sds, smoothing),
                smooth(the_means + the_sds, smoothing),
                color=colors[i],
                alpha=0.2,
            )
        for ax in [ax1, ax2, ax3]:
            ax.legend()
            ax.set_ylim(0, self.max_steps)
        ax1.set_title("a) No replay")
        ax2.set_title("b) Replay, batch size=1")
        ax3.set_title("c) Replay, batch size=64")
        plt.savefig(f"{filename}.pdf", bbox_inches ='tight')


if __name__ == "__main__":
    e = EverythingPlotter(experiment_dir="lisa")
    #e.plot_weights(filename="thick_weights")
    e.plot_replay_variations(filename="cartpole-test")
