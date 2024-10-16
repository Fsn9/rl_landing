from collections import deque, namedtuple
from random import sample
import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

DQNInteraction = namedtuple('Interaction', ['cur_state',
                                         'action',
                                         'reward',
                                         'next_state',
                                         'termination'
                                         ])

PPOInteraction = namedtuple('Interaction', ['cur_state',
                                         'action',
                                         'reward',
                                         'log_prob',
                                         'value',
                                         'termination'
                                         ])

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TODO criar class geral Memory

class ReplayMemory:
    def __init__(self, capacity, batch_size = 8):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.memory)

    def store(self, cur_state, action, reward, next_state, termination):
        self.memory.append(DQNInteraction(cur_state,action,reward,next_state,termination))
    
    def sample(self):
        return sample(list(self.memory), self.batch_size)

class RolloutBuffer:
    def __init__(self, capacity, minibatch_size):
        self.memory = deque(maxlen=capacity)
        self.batch_size = capacity 
        self.minibatch_size = minibatch_size
        self.__inds = np.arange(self.batch_size)
    
    def __len__(self):
        return len(self.memory)

    def store(self, cur_state, action, reward, log_prob, value, termination):
        self.memory.append(PPOInteraction(cur_state,action,reward,log_prob,value,termination))
    
    def sample(self):
        return sample(list(self.memory), self.minibatch_size)
    
    def get_shuffled_inds(self):
        np.random.shuffle(self.__inds)
        return self.__inds
    
    """ Return samples of memory deque and clear deque if remove=True """
    def load(self, device, remove=True):
        batch_state = torch.vstack([interaction.cur_state for interaction in self.memory]).to(device).clone()
        batch_action = torch.vstack([interaction.action for interaction in self.memory]).to(device).clone()
        batch_reward = torch.vstack([interaction.reward for interaction in self.memory]).to(device).clone()
        batch_log_prob = torch.vstack([interaction.log_prob for interaction in self.memory]).to(device).clone()
        batch_value = torch.vstack([interaction.value for interaction in self.memory]).to(device).clone()
        batch_termination = torch.vstack([interaction.termination[0] for interaction in self.memory]).to(device).clone()
        if remove:
            self.memory.clear()
        return batch_state, batch_action, batch_reward, batch_log_prob, batch_value, batch_termination