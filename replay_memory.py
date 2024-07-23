from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, cur_state, action, reward, next_state):
        self.memory.append()