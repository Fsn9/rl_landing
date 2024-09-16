import gymnasium as gym
from ros2_msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Range
import numpy as np
import time



def map_to_closest(value, resolution=GRID_RESOL):
    """
    Maps a given value to the nearest multiple of the specified resolution.

    Args:
        value (float): The input value that needs to be mapped.
        resolution (float): The resolution to which the value should be rounded 
                            (default is GRID_RESOL).

    Returns:
        float: The value rounded to the nearest multiple of the resolution.
    """
   
    return float(round(value / resolution) * resolution)
    
def state_changed(prev_state, current_state, action):
    """
    Checks if the state has changed based on the action.

    Args:
        prev_state (list): The previous state of the system (e.g., position or state vector).
        current_state (list): The current state of the system.
        action (int): The action determining the type of state change to check.

    Returns:
        bool: True if the state has changed as per the action, False otherwise.
    """
     
    if action in (0, 1, 2, 3, 4):
        # Check if there is a change in one direction
        return any(prev_state[i] != current_state[i] for i in range(len(current_state)))
    elif  action in (5, 6, 7, 8):
        # Check if there is a change in two directions
        changes = sum(1 for i in range(len(current_state)) if prev_state[i] != current_state[i])
        return changes >= 2
    return False

class CustomEnv(gym.Env):
    """
    Custom Gym Environment for real tests.
    This environment handles drone flight simulation, movement commands, and state updates.
    """
    
    
    def __init__(self):
        """
        Initializes the environment by setting up ROS2 nodes, publishers, subscribers.
        """
        super(CustomEnv, self).__init__()

        # Initialize ROS 2 node
        rclpy.init()
        self.node = rclpy.create_node('custom_env_node')

        #Define observation space (position) and action space (discrete actions)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

        # State attributes
        self.state = np.array([0, 0, 0], dtype=np.float32)
        self.crashed = False
        self.landed = False
        self.step_counter = 0
        self.eps = 0
        self.inside = 0
        self.last_distance = float("inf")

        # Landing pad attributes
        self.landing_pad_position = np.array((0, 0, 0))  # Center position
        self.landing_pad_radius = 12.5  # Radius of the landing pad area

        # Initialize ROS 2 publishers, subscribers, and services
        #self.move_command = GoToXYZ()
        self.move_command = GoToXYZ_vel()
        self.position_subscriber = self.node.create_subscription(Odometry, 'mavros/global_position/local', self.position_callback, qos_profile)
        self.position_subscriber_Z = self.node.create_subscription(Range, 'mavros/rangefinder/rangefinder', self.position_callback_z, qos_profile)
        self.check_armed = CheckArmed()  

        time.sleep(2)

    def position_callback(self, msg):
        
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)
        self.state[:2] = position

    def position_callback_z(self, msg):
        
        position = np.array([msg.range], dtype=np.float32)
        self.state[2] = position[0]-0.19


    def reset(self):
        """
        Resets the environment to an initial state with a random starting position.
        Returns the initial state and an empty info dictionary.
        """

        print("Waiting")
        input()

        rclpy.spin_once(self.node)

        self.state[0] -= 0.00
        self.state[1] -= -0.00
        self.state[2] -= 0.0

        print("First state:", self.state)
        self.state = [map_to_closest(num) for num in self.state]
        
        # Reset internal variables
        self.last_distance = float("inf")
        self.inside = 0
        self.step_counter = 0
        self.eps += 1
        self.crashed = False
        self.landed = False
        self.previous_inside = None
        self.previous_outside = None
        info = {}

        return self.state, info

    def step(self, action):
        """
        Performs a step in the environment by executing the given action.
        Returns the updated state, reward, done flag, and additional info.
        """

        # Take action in the environment
        reward = -50.0  # Placeholder reward
        done = False  # Placeholder termination flag
        info = {}     # Placeholder additional information
        
        # Action-to-direction mapping
        actions = {
            0: 'left', 1: 'right', 2: 'forward', 3: 'backward',
            4: 'down', 5: 'lb', 6: 'lf', 7: 'rb', 8: 'rf'
        }
        direction = actions.get(action)

        # Move the drone in the specified direction
        prev_state = self.state.copy()
        if direction:
            self.move_command.pub_msg(self.state, direction)
        rclpy.spin_once(self.node)
        self.state = [map_to_closest(num) for num in self.state]
        
        rclpy.spin_once(self.node)
        rclpy.spin_once(self.node)

        #offset
        self.state[0] -= 0.00
        self.state[1] -= -0.00
        self.state[2] -= 0.0

        print("ROS State:", self.state)
        distance_to_pad = np.linalg.norm(self.state - self.landing_pad_position)
        
        self.state = [map_to_closest(num) for num in self.state]
        
        # Check if the drone is within the finite 3D space around the landing pad
        position = self.state[:3]
        #distance_to_pad = np.linalg.norm(position - self.landing_pad_position)

        if all(abs(p) < GRID_RESOL for p in position):
            self.landed = True
            done = True
        elif position[2] < GRID_RESOL:
            self.crashed = True
            done = True

        if self.previous_outside is not None:
            if np.sqrt(self.state[0]*self.state[0] + self.state[1]*self.state[1]) < GRID_RESOL:
                if self.inside :
                    shaping = -200 * np.sqrt( 10* self.state[2]**2)
                    reward = shaping - self.previous_inside
                    self.inside = 1
                else:
                    shaping = -200 * np.sqrt(10* self.state[0]**2 + 10* self.state[1]**2 + 1*self.state[2]**2)
                    reward = shaping - self.previous_outside
                    self.inside = 0
            else:
                shaping = -200 * np.sqrt(10* self.state[0]**2 + 10* self.state[1]**2 + 1*self.state[2]**2)
                reward = shaping - self.previous_outside

        self.previous_inside = shaping = -200 * np.sqrt( 10* self.state[2]**2)
        self.previous_outside = -200 * np.sqrt(10* self.state[0]**2 + 10* self.state[1]**2 + 1*self.state[2]**2)

        
        print("Action:", action)
        print("Current state:", self.state)
        print("Distance to landing pad:", distance_to_pad)
        
        if self.landed:
            reward = 400
            done = True
            print("Landed")
        elif self.crashed:
            reward = -200 * distance_to_pad #/ GRID_RESOL
            done = True
            print("Crashed")

        self.step_counter += 1

        print("Reward:", reward)
        
        return self.state, reward, done, done, info
    
    
    def close(self):
         # Clean up ROS 2 resources
        self.node.destroy_node()

        rclpy.shutdown()
