import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from geographic_msgs.msg import GeoPointStamped
from sensor_msgs.msg import Image

from gz.msgs10.pose_pb2 import Pose as PosePB2
from gz.msgs10.boolean_pb2 import Boolean as BooleanPB2
import gz.transport13 as transport13

from rl_landing.networks import OneLayerMLP

from rl_landing.detector.network_modules import VisionTransformer

from rl_landing.replay_buffers import ReplayMemory, RolloutBuffer

from torchvision.transforms import ToTensor, Grayscale, Resize, ToPILImage

import torch

import cv2

import torch.optim as optim
from torch.linalg import norm

import numpy as np

import random

from .action_spaces import *

import csv

import time
import calendar

import wandb

import os

from rl_landing.agent import *
from rl_landing.illegal_actions import IllegalActions
import json

from copy import deepcopy

class SAC(RL):
    def __init__(self, params):
        """ Args disambiguation """
        self.params = params

        self.name = params.get('name')
        self.to_train = params.get('to_train')
        self.to_test = params.get('to_test')
        self.to_resume = params.get('to_resume')
        self.model_path = params.get('model_path')
        self.pkg_path = params.get('pkg_path')
        self.world_ptr = params.get('world_ptr')

        self.best_actor_model_path = self.model_path.replace("best", "best_actor")
        self.best_critic_model_path = self.model_path.replace("best", "best_critic")

        self.last_actor_model_path = self.model_path.replace("best", "last_actor")
        self.last_critic_model_path = self.model_path.replace("best", "last_critic")

        # TODO put these in RL class
        self.alpha = 0.2  # entropy coefficient
        self.gamma = 0.99
        self.tau = 0.005  # for soft update of target networks
        self.lr = 0.0003
        self.batch_size = 8


        self.config = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'lr': self.lr,
            'batch_size': self.batch_size,
        }  # Config for logging (e.g. Wandb)

        super().__init__(params)

        self.action_space = simple_actions
        self.action_space_len = len(simple_actions)

        self.input_size = params.get('input_size')
        self.output_size = self.action_space_len

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Initialize neural nets setup """
        self.actor = OneLayerMLP(self.input_size, self.output_size).to(self.device)
        self.critic_1 = OneLayerMLP(self.input_size, self.output_size).to(self.device)
        self.critic_2 = OneLayerMLP(self.input_size, self.output_size).to(self.device)
        self.target_critic_1 = OneLayerMLP(self.input_size, self.output_size).to(self.device)
        self.target_critic_2 = OneLayerMLP(self.input_size, self.output_size).to(self.device)

        # Copy behaviour policy's weights to target net
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.target_net.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)
        
        self.memory_capacity = 1000
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size)


        # TODO update to SAC networks 
        # """ If test or resume args active, load models """
        # if self.model_path:
        #     self.loaded_model = torch.load(self.model_path, weights_only=False) # This loads object
        #     self.main_net.load_state_dict(self.loaded_model['model_state_dict'])
        #     print(f'Loaded main model ({self.model_path})')

        #     self.loaded_target_model = torch.load(self.best_target_model_path, weights_only=False)
        #     self.target_net.load_state_dict(self.loaded_target_model['model_state_dict'])
        #     print(f'Loaded target model ({self.best_target_model_path})')
            
        #     self.optimizer.load_state_dict(self.loaded_model['optimizer_state_dict'])
        #     print(f'Loaded optimizer model ({self.optimizer})')

        """ Metrics """
        self.counter_steps = 0
        self.counter_episodic_steps = 0
        self.num_episodes = 0
        self.MAX_STEPS = 50
        self.MAX_X = 8
        self.MAX_Y = 8
        self.MAX_Z = 8
        self.cumulative_reward = 0
        self.CRITICAL_HEIGHT_MIN = 0.5
        self.CRITICAL_HEIGHT_MAX = 8
        self.LANDED_ALLOWED_DIAMETER = 1 # TODO should be the diameter of the platform
        self.FIELD_OF_VIEW_THRESHOLD = 1
        self.num_crashes = 0
        self.num_landings = 0
        self.max_reward = -1000

        global timestamp
        if params.get('log'):
            with open(os.path.join(self._run_path, 'metrics_' + timestamp + '.csv'), 'w', newline='') as metrics_file:
                writer = csv.writer(metrics_file)
                writer.writerow(['Avg reward','Num landings','Num crashes'])

        self.state = self.observe()
        print('Initial state:', self.state)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_decay * self.num_episodes + self.epsilon_i, self.epsilon_f)

    def store(self, cur_state, action, reward, next_state, terminated):
        self.memory.store(cur_state, action, reward, next_state, terminated)
    
    """ On end episode decay epsilon and resets landing target position """
    def reset(self):
        self.world_ptr.spin() # spin callbacks to get last UAV position
        
        self.decay_epsilon()

        """ Get position of UAV and assign it to marker """
        position = deepcopy(self.world_ptr.cur_position) # Get last received position of UAV. That will be the new marker position
        position += np.random.rand(3) * NEST_WIDTH # sum some randomness to start not exactly aligned with UAV
        self.world_ptr.reset_marker(position)

        print(f'Reseting episode and new epsilon of {self.epsilon} and new marker position of {position}')
    
    # TODO put observe in class RL
    """
    returns a torch.tensor of the difference between the current pose and the landing target position
    """
    def observe(self):
        if self.world_ptr.cur_position is None: # To contour bug of position callbacks not being rolled
            self.world_ptr.cur_position = INITIAL_POSITION
        return torch.tensor(self.world_ptr.cur_position - self.world_ptr.landing_target_position).float().to(self.device)

    def compute_reward(self, state, action, next_state, termination, prev_pos, next_pos):
        _, reason = termination[0], termination[1]

        prev_dist_to_marker = prev_pos - self.world_ptr.landing_target_position
        next_dist_to_marker = next_pos - self.world_ptr.landing_target_position

        dist = np.abs(prev_dist_to_marker) - np.abs(next_dist_to_marker)

        dx, dy, dz = dist[0], dist[1], dist[2]
        
        print('def compute_reward')
        print('prev dist: ', prev_dist_to_marker)
        print('next dist: ', next_dist_to_marker)
        print('dist:', dist)
        print('dx,dy,dz:',dx,dy,dz)

        if reason == "landed":
            self.num_landings += 1
            return torch.tensor([2], dtype=torch.float32).to(self.device)
        elif reason == "crashed":
            self.num_crashes += 1
            return torch.tensor([-2], dtype=torch.float32).to(self.device)
        else:
            # TODO falta normalizar, ou seja descobrir o MAX_DZ, MAX_DY, MAX_DX
            reward_sum = 0
            reward_sum += 0.33 * (dx > 0) # incentivizes approaching in x
            reward_sum += 0.33 * (dy > 0) # incentivizes approaching in y
            reward_sum += 0.33 * (dz > 0) # incentivizes approaching in z # TODO beneficiar mais altura? (aumentar o ganho kz?)
            self.cumulative_reward += reward_sum
            return torch.tensor([reward_sum], dtype=torch.float32).to(self.device)
    
    def log_metrics(self):
        global timestamp
        """ CSV log """
        avg_reward = (self.cumulative_reward / self.counter_steps).item()
        with open(os.path.join(self._run_path, 'metrics_' + timestamp + '.csv'), 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([avg_reward,self.num_landings,self.num_crashes])
        """ Wandb log """
        if self.params['log']:
            self.run.log({"avg reward": avg_reward, "num_landings": self.num_landings, "num_crashes": self.num_crashes, "epsilon": self.epsilon})
        return avg_reward

    """ Returns if terminated flag and the reason """
    def terminated(self, state, action, next_state):
        x_pos, y_pos, z_pos = self.world_ptr.cur_position[0], self.world_ptr.cur_position[1], self.world_ptr.cur_position[2] # get already updated next positions

        # if max steps
        if self.counter_episodic_steps > self.MAX_STEPS:
            return torch.tensor([True]), "max_steps"
        
        # if out of allowed area
        if abs(x_pos) > self.MAX_X or abs(y_pos) > self.MAX_Y or z_pos > self.MAX_Z or z_pos <= 0:
            return torch.tensor([True]), "outside"

        # if artuga is outside the field of view
        dist_to_marker = np.abs(self.world_ptr.cur_position)[:2] - np.abs(self.world_ptr.landing_target_position)[:2]
        if any(dist_to_marker > self.FIELD_OF_VIEW_THRESHOLD):
            return torch.tensor([True]), "outside_fov"
        
        # if crashed
        if (z_pos < self.CRITICAL_HEIGHT_MIN or z_pos <= 0) and ((abs(state[0]) > self.LANDED_ALLOWED_DIAMETER) or (abs(state[1]) > self.LANDED_ALLOWED_DIAMETER)):
            return torch.tensor([True]), "crashed"
        
        # if landed
        if (z_pos < self.CRITICAL_HEIGHT_MIN) and (abs(state[0]) < self.LANDED_ALLOWED_DIAMETER) and (abs(state[1]) < self.LANDED_ALLOWED_DIAMETER):
            return torch.tensor([True]), "landed"

        return torch.tensor([False]), "none"
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            print('Not enough samples to learn')
            return
        """ 
            Get interactions 
            note: Only states are not detached (because they contain gradients from the detector)
        """
        mini_batch = self.memory.sample()

        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        termination_batch = []
        for interaction in mini_batch:
            state_batch.append(interaction.cur_state)
            action_batch.append(interaction.action)
            next_state_batch.append(interaction.next_state)
            reward_batch.append(interaction.reward)
            termination_batch.append(interaction.termination[0])
        state_batch = torch.stack(state_batch, dim=0).to(self.device) # shape (B,S)
        action_batch = torch.stack(action_batch, dim=0).to(self.device) # shape (B,1)
        reward_batch = torch.stack(reward_batch, dim=0).to(self.device).squeeze(dim=1) # shape (B)
        termination_batch = torch.stack(termination_batch, dim=0).to(self.device).squeeze(dim=1) # shape (B)
        #termination_batch[0] = True # TODO remove
        non_terminated_idxs = (termination_batch == False).nonzero().squeeze(dim=1) # squeeze because it delivers (B,1). I want (B,)

        with torch.no_grad():
            # Compute the target Q-value
            next_actions, next_log_probs = self.actor.sample(next_state_batch)
            target_q1_next = self.target_critic_1(next_state_batch, next_actions)
            target_q2_next = self.target_critic_2(next_state_batch, next_actions)
            target_q_min = torch.min(target_q1_next, target_q2_next) - self.alpha * next_log_probs
            target_q_value = reward_batch + (1 - termination_batch) * self.gamma * target_q_min

        # Update critics
        q1_value = self.critic_1(state_batch, action_batch)
        q2_value = self.critic_2(state_batch, action_batch)
        critic_1_loss = nn.MSELoss()(q1_value, target_q_value)
        critic_2_loss = nn.MSELoss()(q2_value, target_q_value)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(state_batch)
        q1_new = self.critic_1(state_batch, new_actions)
        q2_new = self.critic_2(state_batch, new_actions)
        q_min_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_min_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decide(self, state):
        return self.e_greedy(state, train=True)

    def e_greedy(self, state, train=False):
        if random.random() > self.epsilon:
            self.actor.eval() # This deactivates batchnorm and dropout layers
            with torch.no_grad(): # disables gradient computation (requires_grad=False)
                logits = self.actor(state.unsqueeze(0))
                action_prob = torch.softmax(logits, dim=-1)
                action = torch.multinomial(action_prob, 1).view(1)  
            if train: 
                self.actor.train()
            print('Exploiting')
            return action

        else:
            print('Exploring')
            return torch.randint(0, self.action_space_len, (1,))

    def test(self):
        raise NotImplementedError

    def train(self, state, decision, next_state, prev_pos, next_pos):
        """ Update some metric variables """
        self.counter_steps += 1
        self.counter_episodic_steps += 1

        action = decision

        """ Check termination """
        termination = self.terminated(state, action, next_state)

        """ Get reward """
        reward = self.compute_reward(state, action, next_state, termination, prev_pos, next_pos)

        """ Store experience """
        self.store(state, action, reward, next_state, termination)

        """ Learn """
        self.learn()

        print(f'# EPISODE {self.num_episodes} | step {self.counter_steps}','state: ', state, 'action: ', action, 'next state: ', next_state, 
              'reward: ', reward, 'termination: ', termination, 'landing target', self.world_ptr.landing_target_position,'\n')

        """ If so, reset episode"""
        if termination[0].item(): # If episode ended
            """ Log metrics """
            avg_reward = self.log_metrics() # log avg reward, num crashes, num lands, epsilon

            self.num_episodes += 1 # increment num of episodes
            self.counter_episodic_steps = 0 # reset counter episodic steps

            # TODO update to SAC
            # """ Save last model """
            # """ Save main policy """
            # torch.save({
            # 'episode': self.counter_episodic_steps,
            # 'model_state_dict': self.main_net.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self._last_model_path)

            # """ Save target policy """
            # torch.save({
            # 'episode': self.counter_episodic_steps,
            # 'model_state_dict': self.target_net.state_dict(),
            # }, self._last_target_model_path)

            # """ Save best model if reward is the best """
            # if avg_reward > self.max_reward:
            #     """ Save main policy """
            #     torch.save({
            #     'episode': self.counter_episodic_steps,
            #     'model_state_dict': self.main_net.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            #     }, self._best_model_path)

            #     """ Save target policy """
            #     torch.save({
            #     'episode': self.counter_episodic_steps,
            #     'model_state_dict': self.target_net.state_dict(),
            #     }, self._best_target_model_path)
            #     self.max_reward = avg_reward

            self.reset() # decays epsilon and new landing target
            return (termination[0], termination[1]) # returns termination cause and new marker position
        
        if self.num_episodes == self.max_episodes:
            print('Ended Learning')
            return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position
        
        return (termination[0], termination[1]) # returns termination cause and new marker position # what returns if everything is ok
    
    def __call__(self):
        if self.to_train:
            return self.train()
        else:
            return self.test()
