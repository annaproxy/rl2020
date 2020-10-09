import random


class ReplayMemory:

    def __init__(self, capacity, replay=True):
        self.capacity = capacity
        self.memory = []
        self.replay = replay

    def push(self, transition):

        if len(self.memory) >= self.capacity:
            # remove first entry
            self.memory.pop(0)
        # fill memory
        self.memory.append(transition)

    def sample(self, batch_size):
        if not self.replay:
            return self.sample_latest(batch_size)
        return random.sample(self.memory, batch_size)

    # Sample only from the current episode???
    def sample_latest(self, batch_size, steps=None):
        return random.sample(self.memory[-batch_size:], batch_size)

    def __len__(self):
        return len(self.memory)
