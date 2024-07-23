import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped, PoseStamped
from geographic_msgs.msg import GeoPointStamped

from .networks import *

from .replay_memory import ReplayMemory

import torch

import numpy as np

from random import randint

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
        self.get_logger().info('I heard position.x: "%s"' % msg.pose.position.x)
    
    def gps_cb(self, msg):
        self.cur_gps = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.get_logger().info('I heard gps altitude: "%s"' % msg.position.altitude)

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
        msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
        
        self.i += 1

        self.publisher_.publish(msg)

        self.spin() # Spin callbacks

        self.get_logger().info('Publishing: "%s"' % msg.twist)

class RL:
    def __init__(self):
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

        self.input_size = 3
        self.output_size = 3

        self.main_net = OneLayerMLP(self.input_size,self.output_size)
        
        self.target_net = OneLayerMLP(self.input_size,self.output_size)

        self.epsilon_decay = -self.epsilon_i / self.final_episode_epsilon_decay
        
        self.memory_capacity = 1000
        self.memory = ReplayMemory(self.memory_capacity)

        self.maximum_distance_landing_target = 8
        self.landing_target_position = np.random.randint(0,self.maximum_distance_landing_target,size=3).tolist()

        self.spin() # spin callbacks to get data from topics
        self.state = self.make_state()
        print(self.state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_decay * self.episode_counter + self.epsilon_i, self.epsilon_f)
    
    def store(self, cur_state, action, next_state, reward):
        self.memory.store(cur_state, action, next_state, reward)
    
    """
    returns a torch.tensor of the difference between the current pose and the landing target position
    """
    def make_state(self):
        print(self.landing_target_position)
        print(self.cur_position)
        return torch.tensor(self.cur_position - self.landing_target_position).float()

    def __call__(self):
        """ 1. Observe """
        state = self.make_state()

        """ 2. Act """
        self.actions = self.main_net(state).detach().cpu().numpy() # call network
        self.actions = self.actions.astype(np.float64) # because ros2 accepts only float64 in velocity fields
        super().__call__(self.actions[0], self.actions[1], self.actions[2]) # send actions to ROS simulator

        """ 2. Store experience """
        #self.store(state, action, next_state, reward)

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
