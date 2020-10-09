import gym
import torch
from dqn import QNetwork
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from MemoryReplay import ReplayMemory
from torch import nn
import torch.nn.functional as F


class Trainer:

    def __init__(self, args):
        assert args.memory > args.batch_size, 'memory smaller than the batch size!'

        # hyperparams
        self.discount_factor = args.discount
        self.batch_size = args.batch_size

        # Set gym environment
        self.env = gym.envs.make(args.env)
        self.env.seed(args.seed)

        # get input and output size from env
        x = self.env.reset()
        self.env.reset()
        input_size = len(self.env.state)
        if(args.env == 'Acrobot-v1'):
            input_size = len(x)
        output_size = self.env.action_space.n

        # Init  model
        self.model = QNetwork(input_size, output_size, args.hidden)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # Init policy
        self.epsilon = args.epsilon
        self.policy = EpsilonGreedyPolicy(self.model, self.epsilon, self.env.action_space.n, args.epsilon_cap)

        # Init memory
        self.memory = ReplayMemory(args.memory)

    def train(self, num_episodes):
            
        global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
        episode_durations = []  #
        for i in range(num_episodes):
            state = self.env.reset()

            steps = 0
            while True:
                # get epsilon update
                epsilon = self.policy.get_epsilon(global_steps)
                # update epsilon
                self.policy.set_epsilon(epsilon)
                # increment steps
                global_steps += 1
                steps += 1

                # sample action and store in memory
                action = self.policy.sample_action(state)
                s_next, reward, done, _ = self.env.step(action)
                self.memory.push((state, action, reward, s_next, done))
                loss = self._train_episode()
                state = s_next

                if done:
                    if i % 10 == 0:
                        print("{2} Episode {0} finished after {1} steps"
                              .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    episode_durations.append(steps)
                    break
        return episode_durations

    def _train_episode(self):
        # don't learn without some decent experience
        if len(self.memory) < self.batch_size:
            return None

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state, done = zip(*transitions)

        # convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)[:, None]
        done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

        # compute the q value
        q_val = self._compute_q_vals(state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self._compute_targets(reward, next_state, done, self.discount_factor)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

    def _compute_q_vals(self, states, actions):
        """
        This method returns Q values for given state action pairs.

        Args:
            states: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: Shape: batch_size x 1

        Returns:
            A torch tensor filled with Q values. Shape: batch_size x 1.
        """
        qvals = self.model(states)
        q_sa = qvals.gather(1, actions)
        return q_sa

    def _compute_targets(self, rewards, next_states, dones, discount_factor):
        """
        This method returns targets (values towards which Q-values should move).

        Args:
            rewards: a tensor of actions. Shape: Shape: batch_size x 1
            next_states: a tensor of states. Shape: batch_size x obs_dim
            dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
            discount_factor: discount
        Returns:
            A torch tensor filled with target values. Shape: batch_size x 1.
        """
        q_sp = self.model(next_states)
        maxq, _ = q_sp.max(dim=1, keepdim=True)
        targets = rewards + discount_factor * maxq * (1 - dones.float())
        return targets
