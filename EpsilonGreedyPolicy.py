import torch
import numpy as np
import random


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon

    def sample_action(self, obs):

        assert self.epsilon <= 1.0, 'epsilon>1 is not a valid probability!'
        policy_choice = np.random.choice(('greedy', 'non-greedy'), p=[1.0 - self.epsilon, self.epsilon])

        if policy_choice == 'greedy':
            with torch.no_grad():
                logits = self.Q(torch.tensor(obs).float())
                action = torch.argmax(logits).item()
        elif policy_choice == 'non-greedy':
            action = random.choice([0, 1])
        else:
            raise NotImplementedError

        return action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    @staticmethod
    def get_epsilon(it):
        # prob of taking greedy action
        greedy = 0.0
        if it > 1000:
            # greedy capped at 0.95
            greedy = 0.95
        else:
            # interpolation betw. 0.0 and 0.95
            greedy = it * 0.95 / 1000

        epsilon = 1.0 - greedy
        return epsilon
