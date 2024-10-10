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

from rl_landing.replay_memory import ReplayMemory

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

current_GMT = time.gmtime()
timestamp = str(calendar.timegm(current_GMT))

POSSIBLE_ARTUGA_COORDINATES = [2,6,-2,-6]
INITIAL_POSITION = np.array([2,2,0])
INPUT_SIZE = 120
NEST_WIDTH = 1.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#resize = Resize(size = ACTUAL_SIZE_TUPLE, antialias=True)
to_grayscale = Grayscale()
to_tensor = ToTensor()
black_observation = torch.zeros((1, INPUT_SIZE, INPUT_SIZE)).to(DEVICE) # C,H,W
empty_rgb_observation = torch.zeros((INPUT_SIZE, INPUT_SIZE, 3)).to(DEVICE) # H,W,C
count_frames = 0


"""
class ROS2Controller that has:
    a publisher to /ap/cmd_vel

    a subscriber to /ap/pose/filtered
    a subscriber to /ap/gps_global_origin/filtered
"""
class ROS2Controller(Node):

    def __init__(self, name='ros2 controller'):
        super().__init__(name)
        
        self.name = name
        
        self.publisher_ = self.create_publisher(TwistStamped, '/ap/cmd_vel', 10)

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
    
        """ Subs"""
        self.pose_sub = self.create_subscription(PoseStamped,'/ap/pose/filtered',self.pose_cb,qos_policy)
        self.pose_sub  # prevent unused variable warning
        self.cur_position = None

        self.gps_sub = self.create_subscription(GeoPointStamped,'/ap/gps_global_origin/filtered',self.gps_cb,qos_policy)
        self.gps_sub  # prevent unused variable warning
        self.cur_gps = None

        self.vis_img_sub = self.create_subscription(Image,'/visual_camera/image',self.vis_img_cb,qos_policy)
        self.vis_img_sub  # prevent unused variable warning
        self.cur_vis_data = None
        self.cur_vis_img = black_observation
        self.count_vis_frames = 0

        """ Publishers """
        self.detection_pub = self.create_publisher(Image, 'detection/image', qos_policy)

        self.i = 0
        
        """ ROS2 api node """
        self.node_set_gz_pose = transport13.Node()

        """ Roll callbacks enabling systems to get first data """
        self.spin()

        """ Landing target object """
        self.maximum_distance_landing_target = 8 # TODO remove this
        position = INITIAL_POSITION # TODO this should come from config
        self.reset_marker(position)

        print(f'Initialized ROS2 Controller {self.name}')
    
    def update_marker_position(self, position):
        #self.landing_target_position = [random.choice(POSSIBLE_ARTUGA_COORDINATES),random.choice(POSSIBLE_ARTUGA_COORDINATES), 0.0]
        #self.landing_target_position[2] = 0.0
        self.landing_target_position = position # TODO truncate to [2,6]

    """ Gazebo set model service"""
    def set_gz_model_pose(self, name, pose):
        # Service request message
        req = PosePB2()
        req.name = name
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
            print(f"Service call was successful.")
            print(f"Response: {response}")
        else:
            print("Service call failed.")
    
    def reset_marker(self, position):
        position[-1] = 0.0
        self.update_marker_position(position)
        self.set_gz_model_pose('artuga_0', position)

    def spin(self):
        rclpy.spin_once(self)

    def pose_cb(self, msg):
        self.cur_position = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z]) + INITIAL_POSITION # the sum is to convert from local to global
        # self.get_logger().info('I heard position.x: "%s"' % msg.pose.position)
    
    def gps_cb(self, msg):
        self.cur_gps = np.array([msg.position.latitude,msg.position.longitude,msg.position.altitude])
        # self.get_logger().info('I heard gps altitude: "%s"' % msg.position)
    
    def vis_img_cb(self, msg):
        global count_frames, black_observation, resize
        
        self.count_vis_frames += 1

        self.cur_vis_data = msg.data # Save msg

    def __call__(self, vx, vy, vz):
        msg = TwistStamped()

        msg.header.frame_id = "base_link"

        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        
        self.i += 1

        self.publisher_.publish(msg)

        self.spin() # Spin callbacks

        self.get_logger().info('Publishing: "%s"' % msg.twist)

class RL:
    def __init__(self, params):
        self.name = params.get('name')

        global timestamp
        
        self.cumulative_reward = 0
        
        """ Initialize wandb log platform """
        # if params.get('to_train'): # TODO should be checked the --log arg in arg parser instead
        #     wandb.login()
        #     self.run = wandb.init(
        #         project="rl_landing",
        #         config=self.config,
        #     )

        """ Initialize revelant paths """
        self._pkg_path = params.get('pkg_path')
        self._model_path = params.get('model_path')
        self._run_path = params.get('run_path')

        """ Initializes runs dir if not existent """
        self._runs_path = os.path.join(self._pkg_path,'runs')
        if not os.path.exists(self._runs_path):
            os.mkdir(self._runs_path)

        """ If training, create new run dir """
        if not (params.get('to_test') or params.get('to_resume')):        
            """ Initializes current results dir """
            self._run_path = os.path.join(self._runs_path, "run" + "_" + timestamp)
            os.mkdir(self._run_path)

            """ Initializes model paths """
            self._last_model_path = os.path.join(self._run_path, "last.pth")
            self._best_model_path = os.path.join(self._run_path, "best.pth")
            self._last_target_model_path = os.path.join(self._run_path, "last_target.pth")
            self._best_target_model_path = os.path.join(self._run_path, "best_target.pth")
        else: # test or resume
            """ Initializes current results dir """
            self._run_path = os.path.join(self._runs_path, "run" + "_" + timestamp) # TODO get run id from <model_path>

            """ Initializes model paths """
            self._last_model_path = os.path.join(self._run_path, "last.pth")
            self._best_model_path = os.path.join(self._run_path, "best.pth")
            self._last_target_model_path = os.path.join(self._run_path, "last_target.pth")
            self._best_target_model_path = os.path.join(self._run_path, "best_target.pth")

    def finish(self):
        print('Finishing RL process')
        #self.run.finish() # closes wandb

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

    def decide(self, state):
        raise NotImplementedError

class Lander(ROS2Controller):
    """
    Lander module incorporating:
    a) A transformer-based detector
    b) A RL policy that learns a landing task
    """
    def __init__(self, params):
        super().__init__(params.get('name'))

        self.spin() # make first spin callbacks to get data from topics

        self.freeze = params.get('freeze')

        """ Load config file with model parameters """
        self._config_file = params.get('config_file')
        self._cfg_path = os.path.join(params.get('pkg_path'), 'config', self._config_file)
        self._cfg = {}
        if not os.path.exists(self._cfg_path):
            os.mkdir(self._cfg_path)
        if self._config_file:
            with open(self._cfg_path,'r') as cfg_json:
                self._cfg = json.load(cfg_json)

        params['config'] = self._cfg # Add config object to params dictionary to be passed to models
        
        params['world_ptr'] = self # Pointer to ros2 and gazebo API such that models can get updated observations

        self.detector = VisionTransformer(params)
        params['input_size'] = self.detector.flatten_size # TODO change this to @property

        self.agent = DQN(params)

        print('Global parameters: ', params)

        self.params = params # Save params
    
    def __call__(self):
        print('Lander.__call__')

        pos = self.cur_position # save current position of agent in the world (global knowledge) for the reward function
        print('UAV position: ', self.cur_position)

        """ 1. Detector """
        obs = Detector.observe(self.cur_vis_data)
        embeddings = self.detector(obs, Lander_Class=True)
        print('Embeddings shape: ', embeddings.shape)

        """ 2. Agent """
        action = self.agent.decide(state=embeddings)

        """ 3. Lander acts in environment with agent's actions """
        self.act(*self.agent.action_space[action.item()])

        """ 4. Wait for state change and save next state"""
        state_start_time = time.time()
        while True:
            self.spin() # roll callbacks to update data observed (TODO: how to ensure rolling all callbacks?)

            """ Get next state """
            obs = Detector.observe(self.cur_vis_data)
            with torch.no_grad(): # Does not require gradients to observe next state
                next_embeddings = self.detector(obs, Lander_Class=True)

            next_pos = self.cur_position # save next position of agent in the world (global knowledge) for the reward function

            distance_between_states = norm(embeddings - next_embeddings) # TODO i could do diff between embs or images
            
            """ If distance covered is bigger than 0.25 or transition duration above 5 seconds """
            if distance_between_states >= 0.25 or (time.time() - state_start_time) >= 5:
                print('norm between states: ', distance_between_states)
                break
        
        """ 5. Train the agent """
        return self.agent.train(state=embeddings,action=action,next_state=next_embeddings,prev_pos=pos,next_pos=next_pos)

    def act(self, *action):
        super().__call__(*action) # Performs the actions in the inherited environment

    def forward(self, x):
        x = self.detector(x, Lander_Class=True)
        action = self.agent(x)

    """ Resets the system. It is the same as resetting the agent """
    def reset(self):
        self.agent.reset()

    def finish(self):
        self.agent.finish()

class Detector(ROS2Controller):
    def __init__(self, params):
        """ Args disambiguation """
        self.params = params
        self.name = self.params['name']
        super().__init__(self.name)

        # self.to_train = train
        # self.to_test = test
        # self.to_resume = resume
        # self.model_path = model

        self.spin() # spin callbacks to get data from topics
        self.observe()

        self.detector = VisionTransformer(params).to(DEVICE)
        
        if self.params['model']:
            self.load_weights(self.params['model'])

        print('Initialized detector')

    def load_weights(self, path): # Put in RL class with extra argument to_train: bool. depending on that is model.eval() or model.train()
        model_state_dict = self.detector.state_dict()

        """ Load model """
        new_model = torch.load(path)

        for key1, key2 in zip(new_model['state_dict'], self.detector.state_dict()): 
            model_state_dict[key2] = new_model['state_dict'][key1] # Assign parameters from new model to old model (contouring the Unexpected Key error)

        self.detector.load_state_dict(model_state_dict) # load new assigned state_dict
        self.detector.eval()
        self.detector.to(device)

    def observe(self):
        self.spin()
        np_arr = np.frombuffer(self.cur_vis_data, np.uint8) # convert sensor_msgs/Image to numpy
        # TODO put in [0,1] float
        np_arr = np.reshape(np_arr, (INPUT_SIZE, INPUT_SIZE, 3)) # Reshape it as RGB 160x160 and batch size = 1
        tensor = to_tensor(np.array(np_arr)).to(DEVICE) # Convert to torch.tensor
        tensor = to_grayscale(tensor) # Convert from RGB to grayscale
        img = ToPILImage()(tensor) # TODO remove after debug
        img.save('/home/francisco/Downloads/pil.png')
        # tensor = torch.permute(tensor, (2,0,1)) # change from (H,W,C) to (C,H,W)
        self.cur_vis_img = tensor
    
    @staticmethod
    def observe(cur_vis_data):
        if cur_vis_data is None: # To avoid callbacks not being rolled in the beginning
            return empty_rgb_observation.unsqueeze(0)
        np_arr = np.frombuffer(cur_vis_data, np.uint8) # convert sensor_msgs/Image to numpy
        # TODO put in [0,1] float
        np_arr = np.reshape(np_arr, (INPUT_SIZE, INPUT_SIZE, 3)) # Reshape it as RGB 160x160 and batch size = 1
        tensor = to_tensor(np.array(np_arr)).to(DEVICE) # Convert to torch.tensor
        tensor = to_grayscale(tensor) # Convert from RGB to grayscale

        tensor = torch.cat((black_observation,black_observation,tensor), dim=0) # TODO this fakes R and G channel and makes a transforms a (1,H,W) tensor into a (3,H,W)

        img = ToPILImage()(tensor) # TODO remove after debug
        img.save('/home/francisco/Downloads/pil.png')

        tensor = tensor.unsqueeze(0) # This adds the batch dimension
        # tensor = torch.permute(tensor, (2,0,1)) # change from (H,W,C) to (C,H,W)
        return tensor

    def __call__(self):
        self.observe()
        """ Feed detector with tensor of (1,3,H,W) """
        # from torchvision.io import read_image
        # img_tensor = read_image('/home/francisco/ros2_ws/src/rl_landing/rl_landing/detector/test/im-concat-visual-fail-304_27-07-2023_17-37-09_bag-landing_fadeup_20_07_2023_visual-fail-_png.rf.d8a95bfb7ae3803f7f4fcab66badd877.jpg.png') # read image and convert to tensor
        # img_tensor = img_tensor.float()/255 # normalize TODO será que é preciso?
        # img_tensor = img_tensor.unsqueeze(0) # add batch dimension
        # img_tensor = img_tensor.to(device) # send to cuda if available
        # print(img_tensor.shape)
        # obs = img_tensor

        obs = torch.cat((black_observation,black_observation,self.cur_vis_img), dim=0) # TODO this fakes R and G channel
        
        """ Fake LiDAR """
        FAKE_LIDAR = True
        if FAKE_LIDAR:
            p_at_l = {'h': (90,110), 'w': (40,60)}# square length artuga
            for i in range(p_at_l['h'][0],p_at_l['h'][1]):
                for j in range(p_at_l['w'][0],p_at_l['w'][1]):
                    if p_at_l['h'][0] + 2 < i < p_at_l['h'][1] - 2 and p_at_l['w'][0] + 2 < j < p_at_l['w'][1] - 2:
                        continue 
                    obs[0,i,j] = 1
        """ Fake thermal """
        FAKE_THERMAL = True
        if FAKE_THERMAL:
            p_at_t = {'h': (94,114), 'w': (44,64)}# square length artuga
            for i in range(p_at_t['h'][0],p_at_t['h'][1]):
                for j in range(p_at_t['w'][0],p_at_t['w'][1]):
                    if p_at_t['h'][0] + 2 < i < p_at_t['h'][1] - 2 and p_at_t['w'][0] + 2 < j < p_at_t['w'][1] - 2:
                        continue 
                    obs[1,i,j] = 1
        obs = obs.unsqueeze(0) # This adds the batch dimension
        bbox, _, obj = self.detector.forward(obs)
        print('bbox:',bbox)
        print('obj:',torch.sigmoid(obj))
        self.publish(obs.squeeze(0), bbox, obj)
        return (bbox,obj)
    
    def publish(self, image, bbox, objectness):
        # image should be in HWC
        RESIZE = 2 # TODO these two lines should be in config
        IMAGE_SIZE = tuple(np.array([INPUT_SIZE,INPUT_SIZE],dtype=np.uint16) * RESIZE)
        FONT_SCALE = 0.2 * RESIZE

        """ Prepare formats """
        bbox = (bbox.cpu().detach().numpy().squeeze() * image.shape[-1] * RESIZE).astype(np.uint8) # to numpy and shape (4,) and unnormalized
        image = (torch.permute(image,(1,2,0)).cpu().detach().numpy() * 255).astype(np.uint8) # to numpy and shape (H,W,C)
        image = np.ascontiguousarray(image)

        """ Resize for better visualization """
        image = cv2.resize(image, (IMAGE_SIZE[0],IMAGE_SIZE[1]))

        """ Draw bbox and objectness """
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            image = cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            image = cv2.putText(image, 'obj: ' + str(round(torch.sigmoid(objectness).item(),2)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            image = cv2.putText(image, 'no prediction and obj: ' + str(round(torch.sigmoid(objectness).item(),2)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), 1, cv2.LINE_AA)
        
        """ Prepare message """
        output_image_msg = Image()
        output_image_msg.height = image.shape[0]  # Image height (rows)
        output_image_msg.width = image.shape[1]   # Image width (columns)
        output_image_msg.encoding = 'rgb8'  # or 'bgr8', 'mono8' depending on your tensor data
        output_image_msg.is_bigendian = False
        output_image_msg.step = image.shape[1] * 3  # Full row length in bytes (width * channels)
        
        """ Convert the NumPy array into a flattened list for ROS message """
        output_image_msg.data = image.tobytes()

        """ Publish """
        self.detection_pub.publish(output_image_msg)

"""
DQN method
"""
class DQN(RL):
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

        self.best_target_model_path = self.model_path.replace("best", "best_target")
        self.last_target_model_path = self.model_path.replace("best", "last_target")

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

        super().__init__(params)

        self.action_space = simple_actions
        self.action_space_len = len(simple_actions)

        self.input_size = params.get('input_size')
        self.output_size = self.action_space_len

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Initialize neural nets setup """
        self.main_net = OneLayerMLP(self.input_size,self.output_size).to(self.device)
        self.target_net = OneLayerMLP(self.input_size,self.output_size).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict()) # Copy behaviour policy's weights to target net
        self.target_net.eval()

        self.tau = 0.001
        self.optimizer = optim.AdamW(self.main_net.parameters(), lr=self.alpha, amsgrad=True)
        self.criterion = nn.MSELoss()

        self.epsilon_decay = -self.epsilon_i / self.final_episode_epsilon_decay
        
        self.memory_capacity = 1000
        self.batch_size = 1
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size)

        """ If test or resume args active, load models """
        if self.model_path:
            self.loaded_model = torch.load(self.model_path, weights_only=False) # This loads object
            self.main_net.load_state_dict(self.loaded_model['model_state_dict'])
            print(f'Loaded main model ({self.model_path})')

            self.loaded_target_model = torch.load(self.best_target_model_path, weights_only=False)
            self.target_net.load_state_dict(self.loaded_target_model['model_state_dict'])
            print(f'Loaded target model ({self.best_target_model_path})')
            
            self.optimizer.load_state_dict(self.loaded_model['optimizer_state_dict'])
            print(f'Loaded optimizer model ({self.optimizer})')

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
            return torch.tensor(2, dtype=torch.float32).to(self.device)
        elif reason == "crashed":
            self.num_crashes += 1
            return torch.tensor(-2, dtype=torch.float32).to(self.device)
        else:
            # TODO falta normalizar, ou seja descobrir o MAX_DZ, MAX_DY, MAX_DX
            reward_sum = 0
            reward_sum += 0.33 * (dx > 0) # incentivizes approaching in x
            reward_sum += 0.33 * (dy > 0) # incentivizes approaching in y
            reward_sum += 0.33 * (dz > 0) # incentivizes approaching in z # TODO beneficiar mais altura? (aumentar o ganho kz?)
            self.cumulative_reward += reward_sum
            return torch.tensor(reward_sum, dtype=torch.float32).to(self.device)
    
    def log_metrics(self):
        global timestamp
        """ CSV log """
        avg_reward = (self.cumulative_reward / self.counter_steps).item()
        with open(os.path.join(self._run_path, 'metrics_' + timestamp + '.csv'), 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([avg_reward,self.num_landings,self.num_crashes])
        """ Wandb log """
        #self.run.log({"avg reward": avg_reward, "num_landings": self.num_landings, "num_crashes": self.num_crashes, "epsilon": self.epsilon})
        return avg_reward

    """ Returns if terminated flag and the reason """
    def terminated(self, state, action, next_state):
        x_pos, y_pos, z_pos = self.world_ptr.cur_position[0], self.world_ptr.cur_position[1], self.world_ptr.cur_position[2] # get already updated next positions

        # if max steps
        if self.counter_episodic_steps > self.MAX_STEPS:
            return torch.tensor(True), "max_steps"
        
        # if out of allowed area
        if abs(x_pos) > self.MAX_X or abs(y_pos) > self.MAX_Y or z_pos > self.MAX_Z or z_pos <= 0:
            return torch.tensor(True), "outside"

        # if artuga is outside the field of view
        dist_to_marker = np.abs(self.world_ptr.cur_position)[:2] - np.abs(self.world_ptr.landing_target_position)[:2]
        if any(dist_to_marker > self.FIELD_OF_VIEW_THRESHOLD):
            return torch.tensor(True), "outside_fov"
        
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
        non_terminated_idxs = (termination_batch == False).nonzero().squeeze(dim=1) # squeeze because it delivers (B,1). I want (B,)

        self.main_net.train()

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
        loss.backward(retain_graph=True)

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

    def decide(self, state):
        return self.e_greedy(state, train=True)

    def greedy(self, state, train=False):
        self.main_net.eval()
        with torch.no_grad(): # disables gradient computation (requires_grad=False)
            y = torch.argmax(self.main_net(state).detach().cpu().unsqueeze(0), axis = 1) # get argmax. action is in shape [1]
        if train: 
            self.main_net.train()
        return y

    def e_greedy(self, state, train=False):
        self.main_net.eval() # This deactivates batchnorm and dropout layers
        if random.random() > self.epsilon: # exploit
            print('Exploiting')
            with torch.no_grad(): # disables gradient computation (requires_grad=False)
                y = torch.argmax(self.main_net(state).detach().cpu().unsqueeze(0), axis = 1) # get argmax. action is in shape [1]
            if train: 
                self.main_net.train()
            return y

        else: # explore
            print('Exploring')
            return torch.randint(0, self.action_space_len, (1,))

    # def forward(self, x, train=False):
    #     self.main_net.eval()
    #     with torch.no_grad(): # disables gradient computation (requires_grad=False)
    #         y = torch.argmax(self.main_net(x).detach().cpu().unsqueeze(0), axis = 1) # get argmax. action is in shape [1]
    #     if train: 
    #         self.main_net.train()
    #     return y

    def test(self):
        # TODO metrics of test
        """ 2. Act"""
        # action = self.greedy(state)
        super().__call__(*self.action_space[action.item()]) # send actions to ROS simulator
        return

    def train(self, state, action, next_state, prev_pos, next_pos):
        """ Update some metric variables """
        self.counter_steps += 1
        self.counter_episodic_steps += 1

        """ Check termination """
        termination = self.terminated(state, action, next_state)

        """ 5. Get reward """
        reward = self.compute_reward(state, action, next_state, termination, prev_pos, next_pos)

        """ 6. Store experience """
        self.store(state, action, reward, next_state, termination)

        """ 7. Learn """
        self.learn()

        print(f'# EPISODE {self.num_episodes} | step {self.counter_steps}','state: ', state, 'action: ', action, 'next state: ', next_state, 
              'reward: ', reward, 'termination: ', termination, 'landing target', self.world_ptr.landing_target_position,'\n')

        """ 9. If so, reset episode"""
        if termination[0].item(): # If episode ended
            """ 9.1. Log metrics """
            avg_reward = self.log_metrics() # log avg reward, num crashes, num lands, epsilon

            self.num_episodes += 1 # increment num of episodes
            self.counter_episodic_steps = 0 # reset counter episodic steps

            """ 9.2. Save last model """
            """ Save main policy """
            torch.save({
            'episode': self.counter_episodic_steps,
            'model_state_dict': self.main_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self._last_model_path)

            """ Save target policy """
            torch.save({
            'episode': self.counter_episodic_steps,
            'model_state_dict': self.target_net.state_dict(),
            }, self._last_target_model_path)

            """ 9.3. Save best model if reward is the best """
            if avg_reward > self.max_reward:
                """ Save main policy """
                torch.save({
                'episode': self.counter_episodic_steps,
                'model_state_dict': self.main_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self._best_model_path)

                """ Save target policy """
                torch.save({
                'episode': self.counter_episodic_steps,
                'model_state_dict': self.target_net.state_dict(),
                }, self._best_target_model_path)
                self.max_reward = avg_reward

            self.world_ptr.spin() # last call to callbacks

            return (termination[0], termination[1]) # returns termination cause and new marker position
        
        if self.num_episodes == self.max_episodes:
            print('Ended Learning')
            # TODO launch here the test after training
            return (termination[0], termination[1]) # returns termination cause and new marker position
        
        return (termination[0], termination[1]) # returns termination cause and new marker position # what returns if everything is ok
    
    def __call__(self,x): # passar args aqui
        if self.to_train:
            return self.train(x)
        else:
            return self.test(x)

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
    def __init__(self, system_name, train, test, resume, model):
        super().__init__('dummy')

"""
Main that inits and returns controller to mavlink module
"""
def main(**kwargs):
    rclpy.init(args=None)

    input_size = 19200 # TODO this should be given in

    if kwargs['mission'] == "detection":
        system = Detector(kwargs)
    else:
        system = list_controllers[kwargs['name']](kwargs)

    return system

list_controllers = {
   'dummy': Dummy,
   'dqn': DQN,
   'lander': Lander,
}

load_system = main # alias for main

if __name__ == '__main__':
    main()
