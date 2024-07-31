from collections import deque
from random import sample
from collections import namedtuple

Interaction = namedtuple('Interaction', ['cur_state',
                                         'action',
                                         'reward',
                                         'next_state',
                                         'termination'
                                         ])

class ReplayMemory:
    def __init__(self, capacity, batch_size = 8):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.memory)

    def store(self, cur_state, action, reward, next_state, termination):
        self.memory.append(Interaction(cur_state,action,reward,next_state,termination))
    
    def sample(self):
        return sample(list(self.memory), self.batch_size)