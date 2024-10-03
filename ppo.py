import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from geographic_msgs.msg import GeoPointStamped

from gz.msgs10.pose_pb2 import Pose as PosePB2
from gz.msgs10.boolean_pb2 import Boolean as BooleanPB2
import gz.transport13 as transport13

from networks import OneLayerMLP

from replay_memory import ReplayMemory

import torch

import torch.optim as optim
from torch.linalg import norm

import numpy as np

import random

from action_spaces import *

import csv

import time
import calendar

import wandb

import os

from agent import *
from illegal_actions import IllegalActions

from controller import RL


class PPO(RL):
    def __init__(self, input_size, controller_name, train, test, resume, model): # *args comes with (controller_name: str, train: bool, test: bool, resume: bool, model: str)
        """ Args disambiguation """
        print('Args given: ', input_size, controller_name, train, test, resume, model)
        self.name = controller_name
        self.to_train = train
        self.to_test = test
        self.to_resume = resume
        self.model_path = model
        self.best_target_model_path = self.model_path.replace("best", "best_target")
        self.last_target_model_path = self.model_path.replace("best", "last_target")

        # PPO specific parameters
        self.gamma = 0.99
        self.lam = 0.95  # GAE (Generalized Advantage Estimation) lambda
        self.learning_rate = 3e-4
        self.clip_ratio = 0.2
        self.update_epochs = 10  # number of policy updates per batch
        self.batch_size = 32
        self.minibatch_size = 8
        self.entropy_coeff = 0.01

        self.config = {
            'gamma': self.gamma,
            'lam': self.lam,
            'learning_rate': self.learning_rate,
            'clip_ratio': self.clip_ratio,
            'update_epochs': self.update_epochs,
            'batch_size': self.batch_size,
            'minibatch_size': self.minibatch_size,
            'entropy_coeff': self.entropy_coeff,
        } # This will be the config in wandb init

        super().__init__(self.name)

        self.action_space = simple_actions
        self.action_space_len = len(simple_actions)

        self.input_size = input_size
        self.output_size = self.action_space_len

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Initialize neural nets setup """
        self.policy_net = OneLayerMLP(self.input_size, self.output_size).to(self.device)
        self.value_net = OneLayerMLP(self.input_size, 1).to(self.device)
        self.optimizer = optim.AdamW(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=self.learning_rate)


        self.memory = ReplayMemory(self.batch_size, self.minibatch_size)

        """ If test or resume args active, load models """
        # if self.model_path:
        #     self.loaded_model = torch.load(self.model_path, weights_only=False) # This loads object
        #     self.policy_net.load_state_dict(self.loaded_model['model_state_dict'])
        #     print(f'Loaded main model ({self.model_path})')

        #     self.loaded_target_model = torch.load(self.best_target_model_path, weights_only=False)
        #     self.value_net.load_state_dict(self.loaded_target_model['model_state_dict'])
        #     print(f'Loaded target model ({self.best_target_model_path})')
            
        #     self.optimizer.load_state_dict(self.loaded_model['optimizer_state_dict'])
        #     print(f'Loaded optimizer model ({self.optimizer})')

        #     exit()

        #     self.loaded_optimizer = torch.load(self.model_path, weights_only=True)

        # if self.to_test:
        #     self.policy_net.load_state_dict(self.loaded_model['model_state_dict'])
        #     print(f'Loaded main model ({self.model_path}) to test')
        # if self.to_resume:
        #     self.policy_net.load_state_dict(self.loaded_model['model_state_dict'])
        #     print(f'Loaded main model ({self.model_path}) to resume')
        #     self.value_net.load_state_dict(self.loaded_target_model['model_state_dict'])
        #     print(f'Loaded target model ({self.target_model_path}) to resume')
        #     self.optimizer.load_state_dict(self.loaded_model['optimizer_state_dict'])
        #     print(f'Loaded optimizer model ({self.optimizer}) to resume')

        # TODO: this should be detached from DQN class
        self.maximum_distance_landing_target = 8
        self.landing_target_position = np.random.randint(0,self.maximum_distance_landing_target,size=3).astype(float).tolist()
        self.landing_target_position[2] = 0.0

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
        self.num_crashes = 0
        self.num_landings = 0
        self.max_reward = -1000

        global timestamp
        # with open(os.path.join(self._log_path, 'metrics_' + timestamp + '.csv'), 'w', newline='') as metrics_file:
        #     writer = csv.writer(metrics_file)
        #     writer.writerow(['Avg reward','Num landings','Num crashes'])

        self.spin() # spin callbacks to get data from topics
        self.state = self.make_state()
        print('Initial state:', self.state)

        self.reset()

    def store(self, state, action, reward, log_prob, value, done):
        """ Store experience for PPO update """
        self.memory.store(state, action, reward, log_prob, value, done)
    
    """ On end episode decay epsilon and resets landing target position """
    def reset(self):
        #self.memory.reset()
        self.landing_target_position = np.random.randint(-self.maximum_distance_landing_target//2, self.maximum_distance_landing_target//2, size=3).astype(float).tolist()
        self.landing_target_position[2] = 0.0
        #print(f'Reseting episode and new epsilon of {self.epsilon}')
    
    """
    returns a torch.tensor of the difference between the current pose and the landing target position
    """
    def make_state(self):
        print("cur_pos: ", self.cur_position)
        return torch.tensor(self.cur_position - self.landing_target_position).float().to(self.device)

    """ Sample action from policy distribution """
    def select_action(self, state):
        
        state = state.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            logits = self.policy_net(state)
        action_prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_prob, 1).item()  # sample action
        return action, action_prob[:, action].item()
    
    def compute_reward(self, state, action, next_state, termination):
        _, reason = termination[0], termination[1]
        dx, dy, dz = abs(next_state[0]) - abs(state[0]), abs(next_state[1]) - abs(state[1]), abs(next_state[2]) - abs(state[2])
        
        if reason == "landed":
            self.num_landings += 1
            return torch.tensor(2).to(self.device)
        elif reason == "crashed":
            self.num_crashes += 1
            return torch.tensor(-2).to(self.device)
        else:
            # TODO falta normalizar, ou seja descobrir o MAX_DZ, MAX_DY, MAX_DX
            reward_sum = 0
            reward_sum += 0.33 * (1 - 2 * (dx >= 0)) # incentivizes approaching in x
            reward_sum += 0.33 * (1 - 2 * (dy >= 0)) # incentivizes approaching in y
            reward_sum += 0.33 * (1 - 2 * (dz >= 0)) # incentivizes approaching in z
            self.cumulative_reward += reward_sum
            return reward_sum
    
    def log_metrics(self):
        global timestamp
        """ CSV log """
        avg_reward = (self.cumulative_reward / self.counter_steps).item()
        with open(os.path.join(self._log_path, 'metrics_' + timestamp + '.csv'), 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([avg_reward,self.num_landings,self.num_crashes])
        """ Wandb log """
        self.run.log({"avg reward": avg_reward, "num_landings": self.num_landings, "num_crashes": self.num_crashes, "epsilon": self.epsilon})
        return avg_reward

    """ Returns if terminated flag and the reason """
    def terminated(self, state, action, next_state):
        x_pos, y_pos, z_pos = self.cur_position[0], self.cur_position[1], self.cur_position[2] # get already updated next positions

        # if max steps
        if self.counter_episodic_steps > self.MAX_STEPS:
            return torch.tensor(True), "max_steps"
        
        # if out of allowed area
        if abs(x_pos) > self.MAX_X or abs(y_pos) > self.MAX_Y or z_pos > self.MAX_Z or z_pos <= 0:
            return torch.tensor(True), "outside"
        
        # if crashed
        if (z_pos < self.CRITICAL_HEIGHT_MIN or z_pos <= 0) and ((abs(state[0]) > self.LANDED_ALLOWED_DIAMETER) or (abs(state[1]) > self.LANDED_ALLOWED_DIAMETER)):
            return torch.tensor(True), "crashed"
        
        # if landed
        if (z_pos < self.CRITICAL_HEIGHT_MIN) and (abs(state[0]) < self.LANDED_ALLOWED_DIAMETER) and (abs(state[1]) < self.LANDED_ALLOWED_DIAMETER):
            return torch.tensor(True), "landed"

        return torch.tensor(False), "none"
    
    """ Compute advantage estimates using GAE """
    def compute_advantage(self, rewards, values, next_value, done):
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - done[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - done[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]

        return advantages
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            print('Not enough samples to learn')
            return
        
        for _ in range(self.update_epochs):
            for batch in self.memory.sample():
                states, actions, rewards, log_probs, values, dones = batch
                next_value = self.value_net(states[-1].unsqueeze(0)).item()
                advantages = self.compute_advantage(rewards, values, next_value, dones)
                advantages = torch.tensor(advantages).to(self.device)
                returns = advantages + values

                # Policy loss
                new_log_probs = torch.log_softmax(self.policy_net(states), dim=-1)
                new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(self.value_net(states).squeeze(), returns)

                # Entropy bonus
                entropy = -torch.mean(torch.sum(torch.softmax(new_log_probs, dim=-1) * new_log_probs, dim=-1))
                loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def test(self):
        raise NotImplementedError

    def train(self, state_detector):
        """ PPO training loop """
        self.counter_steps += 1
        self.counter_episodic_steps += 1

        """ 1. Observe """
        state = self.make_state()

        """ 2. Act """
        action, log_prob = self.select_action(state)
        value = self.value_net(state).item()
        super().__call__(*self.action_space[action])  # send action to ROS simulator

        """ Get state until state changes """
        state_start_time = time.time() 
        # print('Before spin: ', state, 'action: ', action)
        while True:
            """ 3. Spin again to get next position from callbacks """
            self.spin() # TODO this should wait for change of state

            """ 4. Get next state """
            next_state = self.make_state()
            distance_between_states = norm(state - next_state)
            
            """ If distance covered is bigger than 0.1 or transition duration above 5 seconds """
            if distance_between_states >= 0.25 or (time.time() - state_start_time) >= 5:
                break
        
        """ Check termination """
        termination = self.terminated(state, action, next_state)[0].item()
        
        """ 5. Get reward """
        reward = self.compute_reward(state, action, next_state, self.terminated(state, action, next_state))
        
        """ 6. Store experience """
        self.store(state, action, reward, log_prob, value, termination)

        """ 7. Learn """
        self.learn()

        print(f'# EPISODE {self.num_episodes} | step {self.counter_steps}')
        print('state: ', state, 'action: ', action, 'next state: ', next_state, 
              'reward: ', reward, 'termination: ', termination, 'landing target', self.landing_target_position,'\n')

        """ 9. If so, reset episode"""
        if termination:
            """ 9.1. Log metrics """
            avg_reward = self.log_metrics() # log avg reward, num crashes, num lands, epsilon

            self.num_episodes += 1 # increment num of episodes
            self.counter_episodic_steps = 0 # reset counter episodic steps

            """ 9.2. Save last model """
            """ Save main policy """
            torch.save({
            'episode': self.counter_episodic_steps,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self._last_model_path)

            """ Save target policy """
            torch.save({
            'episode': self.counter_episodic_steps,
            'model_state_dict': self.value_net.state_dict(),
            }, self._last_target_model_path)

            """ 9.3. Save best model if reward is the best """
            if avg_reward > self.max_reward:
                """ Save main policy """
                torch.save({
                'episode': self.counter_episodic_steps,
                'model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self._best_model_path)

                """ Save target policy """
                torch.save({
                'episode': self.counter_episodic_steps,
                'model_state_dict': self.value_net.state_dict(),
                }, self._best_target_model_path)
                self.max_reward = avg_reward

            self.reset() # decays epsilon and new landing target
            return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position

        if self.num_episodes == self.max_episodes:
            print('Ended Learning')
            return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position
        
        return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position # what returns if everything is ok
    
    def __call__(self):
        if self.to_train:
            return self.train()
        else:
            return self.test()
