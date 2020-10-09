import torch
import numpy as np
import random
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, action_space, cap_it):
        self.Q = Q
        self.epsilon = epsilon
        self.action_space = action_space
        self.cap_it = cap_it

    def sample_action(self, obs):

        assert self.epsilon <= 1.0, 'epsilon>1 is not a valid probability!'
        policy_choice = np.random.choice(('greedy', 'non-greedy'), p=[1.0 - self.epsilon, self.epsilon])

        if policy_choice == 'greedy':
            with torch.no_grad():
                logits = self.Q(torch.tensor(obs).float().to(device))
                action = torch.argmax(logits).item()
        elif policy_choice == 'non-greedy':
            action = random.choice(list(range(self.action_space)))
        else:
            raise NotImplementedError

        return action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self, it):
        # prob of taking greedy action
        greedy = 0.0
        if it > self.cap_it:
            # greedy capped at 0.95
            greedy = 0.95
        else:
            # interpolation betw. 0.0 and 0.95
            greedy = it * 0.95 / self.cap_it

        epsilon = 1.0 - greedy
        return epsilon
