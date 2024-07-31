import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped, PoseStamped
from geographic_msgs.msg import GeoPointStamped

from .networks import *

from .replay_memory import ReplayMemory

import torch

import torch.optim as optim

import numpy as np

from random import randint, random

from .action_spaces import *

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
    
    def spin(self):
        rclpy.spin_once(self)

    def pose_cb(self, msg):
        self.cur_position = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.get_logger().info('I heard position.x: "%s"' % msg.pose.position)
    
    def gps_cb(self, msg):
        self.cur_gps = np.array([msg.position.latitude,msg.position.longitude,msg.position.altitude])
        self.get_logger().info('I heard gps altitude: "%s"' % msg.position)

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

class RL:
    def __init__(self):
        self.cumulative_reward = 0
        pass

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

"""
DQN method
"""
class DQN(ROS2Controller):
    def __init__(self):
        # initalizes DQN offline or online
        ## if offline, does not initalize ROS stuff because it will learn from offline data
        super().__init__('dqn')

        self.alpha = 0.00025
        self.gamma = 0.99
        self.epsilon_i = 1
        self.epsilon_f = 0.05
        self.epsilon = self.epsilon_i
        self.max_episodes = 1000
        self.episode_counter = 0
        self.final_episode_epsilon_decay = int(0.9 * self.max_episodes) # on this episode, epsilon stays constant until the end

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

        self.maximum_distance_landing_target = 8
        self.landing_target_position = np.random.randint(0,self.maximum_distance_landing_target,size=3).tolist()

        """ Metrics """
        self.counter_steps = 0
        self.MAX_STEPS = 30
        self.MAX_X = 8
        self.MAX_Y = 8
        self.MAX_Z = 8
        self.cumulative_reward = 0
        self.CRITICAL_HEIGHT_MIN = 1
        self.CRITICAL_HEIGHT_MAX = 8
        self.LANDED_ALLOWED_DIAMETER = 1 # TODO should be the diameter of the platform

        self.spin() # spin callbacks to get data from topics
        self.state = self.make_state()
        print(self.state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_decay * self.episode_counter + self.epsilon_i, self.epsilon_f)
    
    def store(self, cur_state, action, reward, next_state, terminated):
        self.memory.store(cur_state, action, reward, next_state, terminated)
    
    """
    returns a torch.tensor of the difference between the current pose and the landing target position
    """
    def make_state(self):
        return torch.tensor(self.cur_position - self.landing_target_position).float()
    
    def e_greedy(self, state):
        if random() > self.epsilon: # exploit
            print('Exploiting')
            with torch.no_grad(): # disables gradient computation (requires_grad=False)
                return torch.argmax(self.main_net(state).detach().cpu().numpy(), axis = 1)[0] # get argmax

        else: # explore
            print('Exploring')
            return torch.randint(0, self.action_space_len, (1,))
    
    def compute_reward(self, state, action, next_state, termination):
        _, reason = termination[0], termination[1]
        dx, dy, dz = abs(next_state[0]) - abs(state[0]), abs(next_state[1]) - abs(state[1]), abs(next_state[2]) - abs(state[2])
        
        if reason == "landed":
            return 2
        elif reason == "crashed":
            return -2
        else:
            # TODO falta normalizar, ou seja descobrir o MAX_DZ, MAX_DY, MAX_DX
            reward_sum = 0
            reward_sum += 0.33 * (1 - 2 * (dx >= 0)) # incentivizes approaching in x
            reward_sum += 0.33 * (1 - 2 * (dy >= 0)) # incentivizes approaching in y
            reward_sum += 0.33 * (1 - 2 * (dz >= 0)) # incentivizes approaching in z
            return reward_sum
    
    """ Returns if terminated flag and the reason """
    def terminated(self, state, action, next_state):
        x_pos, y_pos, z_pos = self.cur_position[0], self.cur_position[1], self.cur_position[2] # get positions

        # if max steps
        if self.counter_steps > self.MAX_STEPS:
            return torch.tensor(True), "max_steps"
        
        # if out of allowed area
        if x_pos > self.MAX_X or y_pos > self.MAX_Y or z_pos > self.MAX_Z:
            return torch.tensor(True), "outside"
        
        # if crashed
        if z_pos < self.CRITICAL_HEIGHT_MIN and x_pos > self.LANDED_ALLOWED_DIAMETER and y_pos > self.LANDED_ALLOWED_DIAMETER:
            return torch.tensor(True), "crashed"
        
        # if landed
        if z_pos < self.CRITICAL_HEIGHT_MIN and x_pos < self.LANDED_ALLOWED_DIAMETER and y_pos < self.LANDED_ALLOWED_DIAMETER:
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
        termination_batch[0] = True # TODO remove
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

        """ 1. Observe """
        state = self.make_state()

        """ 2. Act """
        action = self.e_greedy(state)
        super().__call__(*self.action_space[action.item()]) # send actions to ROS simulator

        """ 3. Spin again to get next state """
        self.spin()

        """ 4. Get next state """
        next_state = self.make_state()

        """ Check termination """
        termination = self.terminated(state, action, next_state)

        """ 5. Get reward """
        reward = self.compute_reward(state, action, next_state, termination)

        """ 6. Store experience """
        self.store(state, action, reward, next_state, termination)

        """ Update some metric values """
        self.cumulative_reward += reward

        """ 7. Learn """
        self.learn()

        print('state: ', state)
        print('action: ', action)
        print('next state: ', next_state)
        print('reward: ', reward)
        print('termination: ', termination)
        print()

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
