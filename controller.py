import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from geographic_msgs.msg import GeoPointStamped

from gz.msgs10.pose_pb2 import Pose as PosePB2
from gz.msgs10.boolean_pb2 import Boolean as BooleanPB2
import gz.transport13 as transport13

from .networks import *

from .replay_memory import ReplayMemory

import torch

import torch.optim as optim
from torch.linalg import norm

import numpy as np

from random import randint, random

from .action_spaces import *

import csv

import time
import calendar

import wandb

current_GMT = time.gmtime()
timestamp = str(calendar.timegm(current_GMT))

"""
class ROS2Controller that has:
    a publisher to /ap/cmd_vel

    a subscriber to /ap/pose/filtered
    a subscriber to /ap/gps_global_origin/filtered
"""
class ROS2Controller(Node):

    def __init__(self, name='ros2 controller'):
        super().__init__(name)
        self.publisher_ = self.create_publisher(TwistStamped, '/ap/cmd_vel', 10)

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
    
        self.pose_sub = self.create_subscription(PoseStamped,'/ap/pose/filtered',self.pose_cb,qos_policy)
        self.pose_sub  # prevent unused variable warning
        self.cur_position = None

        self.gps_sub = self.create_subscription(GeoPointStamped,'/ap/gps_global_origin/filtered',self.gps_cb,qos_policy)
        self.gps_sub  # prevent unused variable warning
        self.cur_gps = None

        self.i = 0
        
        self.node_set_gz_pose = transport13.Node()
        
    """ Gazebo set model service"""
    def set_gz_model_pose(self, name, pose):
        # Service request message
        req = PosePB2()
        req.name = "artuga_0"
        req.position.x = pose[0]
        req.position.y = pose[1]
        req.position.z = pose[2]
        req.orientation.w = 1.0
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = 0.0

        result, response = self.node_set_gz_pose.request("/world/map/set_pose", req, PosePB2, BooleanPB2, timeout=10)

        print(f'Marker set to position:\n{req.position}')
        
        if result:
            print(f"Service call was successful. {result}")
            print(f"Response: {response}")
        else:
            print("Service call failed.")
    
    def spin(self):
        rclpy.spin_once(self)

    def pose_cb(self, msg):
        self.cur_position = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        # self.get_logger().info('I heard position.x: "%s"' % msg.pose.position)
    
    def gps_cb(self, msg):
        self.cur_gps = np.array([msg.position.latitude,msg.position.longitude,msg.position.altitude])
        # self.get_logger().info('I heard gps altitude: "%s"' % msg.position)

    # def action_pub(self):
    #     msg = TwistStamped()
    #     msg.header.frame_id = "base_link"
    #     msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
    #     self.publisher_.publish(msg)
    #     self.get_logger().info('Publishing: "%s"' % msg.twist)
    #     self.i += 1

    def __call__(self, vx, vy, vz):
        msg = TwistStamped()

        msg.header.frame_id = "base_link"

        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        #msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
        msg.twist.linear.z = vz
        
        self.i += 1

        self.publisher_.publish(msg)

        self.spin() # Spin callbacks

        self.get_logger().info('Publishing: "%s"' % msg.twist)

class RL(ROS2Controller):
    def __init__(self, name):
        super().__init__('dqn')

        global timestamp
        
        self.cumulative_reward = 0
        
        """ Initialize wandb log platform """
        wandb.login()
        self.run = wandb.init(
            project="rl_landing",
            config=self.config,
        )
    
    def finish(self):
        print('Finishing RL process')
        self.run.finish() # closes wandb

    def learn(self):
        # based on random mini batch, perform gradient step on the policy
        pass

    def act(self):
        # according to e-greedy policy, chooses random or greedy action
        pass

    def store(self):
        # stores transition in replay memory
        pass

    def sample(self):
        # samples batch from replay memory
        pass

    def checkpoint(self):
        # saves checkpoint
        pass

    def reset(self):
        pass

"""
DQN method
"""
class DQN(RL):
    def __init__(self):
        # initalizes DQN offline or online
        ## if offline, does not initalize ROS stuff because it will learn from offline data
        self.alpha = 0.00025
        self.gamma = 0.99
        self.epsilon_i = 1
        self.epsilon_f = 0.05
        self.epsilon = self.epsilon_i
        self.max_episodes = 1000
        self.final_episode_epsilon_decay = int(0.9 * self.max_episodes) # on this episode, epsilon stays constant until the end
        
        self.config = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_i': self.epsilon_i,
            'epsilon_f': self.epsilon_f,
            'max_episodes': self.max_episodes,
            'final_episode_epsilon_decay': self.final_episode_epsilon_decay,
        } # This will be the config in wandb init

        super().__init__('dqn')

        self.action_space = simple_actions
        self.action_space_len = len(simple_actions)

        self.input_size = 3
        self.output_size = self.action_space_len

        """ Initialize neural nets setup """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.main_net = OneLayerMLP(self.input_size,self.output_size).to(self.device)
        self.target_net = OneLayerMLP(self.input_size,self.output_size).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict()) # Copy behaviour policy's weights to target net
        self.tau = 0.001
        self.optimizer = optim.AdamW(self.main_net.parameters(), lr=self.alpha, amsgrad=True)
        self.criterion = nn.MSELoss()

        self.epsilon_decay = -self.epsilon_i / self.final_episode_epsilon_decay
        
        self.memory_capacity = 1000
        self.batch_size = 8
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size)

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

        global timestamp
        with open('metrics_' + timestamp + '.csv', 'w', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(['Avg reward','Num landings','Num crashes'])

        self.spin() # spin callbacks to get data from topics
        self.state = self.make_state()
        print('Initial state:', self.state)

        self.reset()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_decay * self.num_episodes + self.epsilon_i, self.epsilon_f)
    
    def store(self, cur_state, action, reward, next_state, terminated):
        self.memory.store(cur_state, action, reward, next_state, terminated)
    
    """ On end episode decay epsilon and resets landing target position """
    def reset(self):
        self.decay_epsilon()
        self.landing_target_position = np.random.randint(-self.maximum_distance_landing_target//2, self.maximum_distance_landing_target//2, size=3).astype(float).tolist()
        self.landing_target_position[2] = 0.0
        print(f'Reseting episode and new epsilon of {self.epsilon}')
    
    """
    returns a torch.tensor of the difference between the current pose and the landing target position
    """
    def make_state(self):
        return torch.tensor(self.cur_position - self.landing_target_position).float().to(self.device)
    
    def e_greedy(self, state):
        if random() > self.epsilon: # exploit
            print('Exploiting')
            with torch.no_grad(): # disables gradient computation (requires_grad=False)
                return torch.argmax(self.main_net(state).detach().cpu().unsqueeze(0), axis = 1) # get argmax. action is in shape [1]

        else: # explore
            print('Exploring')
            return torch.randint(0, self.action_space_len, (1,))
    
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
        with open('metrics_' + timestamp + '.csv', 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([avg_reward,self.num_landings,self.num_crashes])
        """ Wandb log """
        self.run.log({"avg reward": avg_reward, "num_landings": self.num_landings, "num_crashes": self.num_crashes, "epsilon": self.epsilon})

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
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            print('Not enough samples to learn')
            return
        
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
        state_batch = torch.stack(state_batch, dim=0).to(self.device)
        action_batch = torch.stack(action_batch, dim=0).to(self.device)
        reward_batch = torch.stack(reward_batch, dim=0).to(self.device)
        termination_batch = torch.stack(termination_batch, dim=0).to(self.device)
        #termination_batch[0] = True # TODO remove
        non_terminated_idxs = (termination_batch == False).nonzero().squeeze() # squeeze because it delivers (B,1). I want (B,)

        """ Get Qs """
        # The idea is feeding the *state* from the batch and then select the Q according to the *action* taken
        predicted_qs = self.main_net(state_batch)
        selected_qs = torch.gather(predicted_qs, dim=1, index=action_batch) # selected action values according to batch
        
        """ Get Q targets """
        next_qs = torch.zeros(self.batch_size).to(device=self.device) # max
        with torch.no_grad():
            target_qs = self.target_net(state_batch[non_terminated_idxs]) # this variable is useful to create just for interpretability
            next_qs[non_terminated_idxs] = target_qs.max(1).values # assign max qs to idxs of non terminal states. the others remain equal to zero.

        """ Compute loss """
        # The update is Q = Q + alpha * (r + gamma * max Q - Q)
        td_target = reward_batch + self.gamma * next_qs
        loss = self.criterion(selected_qs, td_target.unsqueeze(1)) # target shape (8) to (8,1)
        self.optimizer.zero_grad()
        loss.backward()

        """ Gradient clipping """
        #torch.nn.utils.clip_grad_value_(self.main_net.parameters(), 100) # clip gradients between 100 and -100
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 100) # clip and normalize gradients for threshold

        """ Update weights """
        self.optimizer.step()

        """ Soft update the target """
        target_net_state_dict = self.target_net.state_dict() # copy
        main_net_state_dict = self.main_net.state_dict() # copy
        for key in main_net_state_dict:
            target_net_state_dict[key] = main_net_state_dict[key] * self.tau + (1 - self.tau) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict) # copy soft updated weights
        
    def __call__(self):
        """ Update some metric variables """
        self.counter_steps += 1
        self.counter_episodic_steps += 1

        """ 1. Observe """
        state = self.make_state()

        """ 2. Act """
        action = self.e_greedy(state)
        super().__call__(*self.action_space[action.item()]) # send actions to ROS simulator

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
        termination = self.terminated(state, action, next_state)

        """ 5. Get reward """
        reward = self.compute_reward(state, action, next_state, termination)

        """ 6. Store experience """
        self.store(state, action, reward, next_state, termination)

        """ 7. Learn """
        self.learn()

        print(f'# EPISODE {self.num_episodes} | step {self.counter_steps}')
        print('state: ', state, 'action: ', action, 'next state: ', next_state, 
              'reward: ', reward, 'termination: ', termination, 'landing target', self.landing_target_position,'\n')

        """ 9. If so, reset episode"""
        if termination[0].item(): # If episode ended
            """ 9.1. Log metrics """
            self.log_metrics() # log avg reward, num crashes, num lands, epsilon

            self.num_episodes += 1 # increment num of episodes
            self.counter_episodic_steps = 0 # reset counter episodic steps

            self.reset() # decays epsilon and new landing target
            return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position
        
        if self.num_episodes == self.max_episodes:
            print('Ended Learning')
            return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position
        
        return (termination[0], termination[1], self.landing_target_position) # returns termination cause and new marker position # what returns if everything is ok

"""
Dummy method
"""
class Dummy(ROS2Controller):
    def __init__(self):
        super().__init__('dummy')

"""
Main that inits and returns controller to mavlink module
"""
def main(args):
    rclpy.init(args=None)

    controller = list_controllers[args]()

    return controller

list_controllers = {
   'dummy': Dummy,
   'dqn': DQN
}

load_controller = main # alias for main

if __name__ == '__main__':
    main()
