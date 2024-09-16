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

import random

from .action_spaces import *

import csv

import time
import calendar

import wandb

from rl_landing.agent import *
from rl_landing.illegal_actions import IllegalActions

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
        if random.random() > self.epsilon: # exploit
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

""" TODO where to put illegal actions """
action_opposition_matrix = np.array([[0,1,0,0,0,0,0,1,1],
                                     [1,0,0,0,0,1,1,0,0],
                                     [0,0,0,1,0,1,0,1,0],
                                     [0,0,1,0,0,0,1,0,1],
                                     [0,0,0,0,0,0,0,0,0],
                                     [0,1,1,0,0,0,0,0,1],
                                     [0,1,0,1,0,0,0,1,0],
                                     [1,0,1,0,0,0,1,0,0],
                                     [1,0,0,1,0,1,0,0,0]]) *1

illegalActions = IllegalActions(action_opposition_matrix)

class dqn(agent_base):

    def __init__(self):
        super().__init__()
        parameters = self.get_default_parameters()
        self.set_parameters(parameters)
        self.in_training = False

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the dqn
        algorithm
        '''
        #
        default_parameters = super().get_parameters()
        #
        # add default parameters specific to the dqn algorithm
        default_parameters['neural_networks']['target_net'] = {}
        default_parameters['neural_networks']['target_net']['layers'] = \
        copy.deepcopy(\
                default_parameters['neural_networks']['policy_net']['layers'])
        #
        #
        # soft update stride for target net:
        default_parameters['target_net_update_stride'] = 1
        # soft update parameter for target net:
        default_parameters['target_net_update_tau'] = 1e-2
        #
        # Parameters for epsilon-greedy policy with epoch-dependent epsilon
        default_parameters['epsilon'] = 1.0 # initial value for epsilon
        default_parameters['epsilon_1'] = 0.1 # final value for epsilon
        default_parameters['d_epsilon'] = 0.00005 # decrease of epsilon
            # after each training epoch
        #
        default_parameters['doubledqn'] = False
        #
        return default_parameters


    def set_parameters(self,parameters):
        #
        super().set_parameters(parameters=parameters)
        #
        ##################################################
        # Use deep Q-learning or double deep Q-learning? #
        ##################################################
        try: # False -> use DQN; True -> use double DQN
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        #
        ##########################################
        # Parameters for updating the target net #
        ##########################################
        try: # after how many training epochs do we update the target net?
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        #
        try: # tau for soft update of target net (value 1 means hard update)
            self.target_net_update_tau = parameters['target_net_update_tau']
            # check if provided parameter is within bounds
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        #
        #
        ########################################
        # Parameters for epsilon-greedy policy #
        ########################################
        try: # probability for random action for epsilon-greedy policy
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try: # final probability for random action during training
            #  for epsilon-greedy policy
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        #
        try: # amount by which epsilon decreases during each training epoch
            #  until the final value self.epsilon_1 is reached
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):
        """
        Use policy net to select an action for the current state

        We use an epsilon-greedy algorithm:
        - With probability epsilon we take a random action (uniformly drawn
          from the finite number of available actions)
        - With probability 1-epsilon we take the optimal action (as predicted
          by the policy net)

        By default epsilon = 0, which means that we actually use the greedy
        algorithm for action selection
        """
        #
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            #
            policy_net = self.neural_networks['policy_net'].to(device)
            #
            with torch.no_grad():
                policy_net.eval()
                #action = policy_net(torch.tensor(state, dtype=torch.float).to(device)).argmax(0).item()
                
                action_scores = policy_net(torch.tensor(state, dtype=torch.float).to(device))
                #print("Q values:", action_scores)
                action_scores = illegalActions.apply(method='policy',action_scores=action_scores)
                sampled_action = action_scores.argmax(0).item()

                policy_net.train()

                illegalActions.update(sampled_action)
                return sampled_action
        else:
            # perform random action
            #return torch.randint(low=0,high=self.n_actions,size=(1,)).item()

            sampled_action = illegalActions.apply(method='random')
            illegalActions.update(sampled_action)

            return sampled_action

    def update_epsilon(self):
        """
        Update epsilon for epsilon-greedy algorithm

        For training we assume that
        epsilon(n) = max{ epsilon_0 - d_epsilon * n ,  epsilon_1 },
        where n is the number of training epochs.

        For epsilon_0 > epsilon_1 the function epsilon(n) is piecewise linear.
        It first decreases from epsilon_0 to epsilon_1 with a slope d_epsilon,
        and then becomes constant at the value epsilon_1.

        This ensures that during the initial phase of training the neural
        network explores more randomly, and in later stages of the training
        follows more the policy learned by the neural net.
        """
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self,epoch):
        """Run one optimization step for the policy net"""
        #
        # if we have less sample transitions than we would draw in an
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch = self.get_samples_from_memory()
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        reward_batch = reward_batch.to(device)
        done_batch = done_batch.to(device)
        #
        policy_net = self.neural_networks['policy_net'].to(device)
        target_net = self.neural_networks['target_net'].to(device)
        #
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net'].to(device)
        #
        policy_net.train() # turn on training mode
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = policy_net(state_batch.to(device=device, dtype=torch.float)).gather(dim=1, index=action_batch.unsqueeze(1))



        # LHS.shape = [batch_size, 1]
        #
        # Evaluate right-hand side of Bellman equation
        if self.doubleDQN:
            # double deep-Q learning paper: https://arxiv.org/abs/1509.06461
            #
            # in double deep Q-learning, we use the policy net for choosing
            # the action on the right-hand side of the Bellman equation. We
            # then use the target net to evaluate the Q-function on the
            # chosen action
            argmax_next_state = policy_net(next_state_batch).argmax(
                                                                    dim=1)
            # argmax_next_state.shape = [batch_size]
            #
            Q_next_state = target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
            # shapes of the various tensor appearing in the previous line:
            # self.target_net(next_state_batch).shape = [batch_size,N_actions]
            # self.target_net(next_state_batch).gather(dim=1,
            #   index=argmax_next_state.unsqueeze(1)).shape = [batch_size, 1]
            # Q_next_state.shape = [batch_size]
        else:
            # in deep Q-learning, we use the target net both for choosing
            # the action on the right-hand side of the Bellman equation, and
            # for evaluating the Q-function on that action
            Q_next_state = target_net(next_state_batch.to(device=device, dtype=torch.float))
            
            #Q learning normalization for this specific case, with unique reward function
            #Q_next_state = (Q_next_state - (-200*np.sqrt(72))) / (400 - (-200*np.sqrt(72))) 
            
            Q_next_state = Q_next_state.max(1)[0].detach()

            #Q_next_state.shape = [batch_size]
            
        #print("Done batch: ",done_batch)
        #print("Reward batch: ",reward_batch)
        #print("Q next state: ",Q_next_state)
        
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        #print("RHS: ",RHS)
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        #
        policy_net.eval() # turn off training mode
        #
        self.update_epsilon() # for epsilon-greedy algorithm
        #
        if epoch % self.target_net_update_stride == 0:
            print("Updating target net")
            self.soft_update_target_net() # soft update target net
        #

    def soft_update_target_net(self):
        """Soft update parameters of target net"""
        #
        # the following code is from https://stackoverflow.com/q/48560227
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)


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
