from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def size(self):
        return len(self.memory)