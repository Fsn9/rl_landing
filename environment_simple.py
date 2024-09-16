import gymnasium as gym
from ros2_msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import subprocess
import os
import signal
import numpy as np
import time
import random

# Create an empty dictionary to store counts (s,a)
counter_dict = {}

# Incrementing counts
def increment_count(key_x, key_y, key_z, key_a):
    if key_x not in counter_dict:
        counter_dict[key_x] = {}
    if key_y not in counter_dict[key_x]:
        counter_dict[key_x][key_y] = {}
    if key_z not in counter_dict[key_x][key_y]:
        counter_dict[key_x][key_y][key_z] = {}
    if key_a in counter_dict[key_x][key_y][key_z]:
        counter_dict[key_x][key_y][key_z][key_a] += 1
    else:
        counter_dict[key_x][key_y][key_z][key_a] = 1


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
    Custom Gym Environment for a drone simulation using ROS2, Gazebo, and MAVROS.
    This environment handles drone flight simulation, movement commands, and state updates.
    """
    
    def __init__(self):
        """
        Initializes the environment by setting up ROS2 nodes, publishers, subscribers, 
        and launching required external processes like Gazebo and ArduPilot.
        """
        super(CustomEnv, self).__init__()
        
        # Initialize ROS 2 node
        rclpy.init()
        self.node = rclpy.create_node('custom_env_node')

        # Define observation space (position) and action space (discrete actions)
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
        self.move_command = GoToXYZ()
        self.position_subscriber = self.node.create_subscription(Odometry, 'mavros/global_position/local', self.position_callback, qos_profile)
        self.check_armed = CheckArmed()

        # Launch external processes
        self.launch_gazebo()
        self.launch_ardupilot()
        #self.launch_ros2_mavros()

        # Prepare the drone for flight
        time.sleep(20)
        self.prepare_flight()

    def prepare_flight(self):
        """
        Prepares the drone for flight by setting the mode to GUIDED, arming the throttle,
        and taking off to a specified altitude.
        """
        # Set mode to GUIDED
        serv_client_mode = SetModeGuided()
        serv_client_mode.send_request()
        serv_client_mode.destroy_node()
        time.sleep(3)

        # Arm throttle
        serv_client_armthrottle = ArmThrottle()
        serv_client_armthrottle.send_request()
        serv_client_armthrottle.destroy_node()
        time.sleep(3)

        # Take off
        serv_client_takeoff = Takeoff()
        serv_client_takeoff.send_request()
        serv_client_takeoff.destroy_node()
        time.sleep(3)

        # Set initial position at altitude
        self.set_init_pos = SetPosition()
        self.set_init_pos.pub_msg([0.0, 0.0, 5.0])
        time.sleep(3)


    def launch_gazebo(self):
        """
        Launches the Gazebo simulation with a predefined world.
        """
        gazebo_command = ["gz", "sim", "-r", "iris_runway.sdf"]
        self.gazebo_process = subprocess.Popen(gazebo_command)

    def launch_ardupilot(self):
        """
        Launches ArduPilot simulation using sim_vehicle.py script.
        """
        ardupilot_command = [
            "bash",
            "-c",
            "cd ~/ros2_ws/src/ardupilot/Tools/autotest/ && ./sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON  --console"
        ]
        self.ardupilot_process = subprocess.Popen(ardupilot_command, preexec_fn=os.setsid)

    def launch_ros2_mavros(self):
        ros2_mavros_command = ["ros2", "launch", "mavros", "apm.launch", "fcu_url:=udp://:14550@"]
        self.mavros_process = subprocess.Popen(ros2_mavros_command, preexec_fn=os.setsid)

    def position_callback(self, msg):
        """
        Callback to update the current position of the drone from ROS Odometry data.
        """
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        self.state[:3] = position

    def reset(self):
        """
        Resets the environment to an initial state with a random starting position.
        Returns the initial state and an empty info dictionary.
        """
        start_pos = [random.randint(-6, 6) + 0.0, random.randint(-6, 6) + 0.0, random.randint(4, 8) + 0.0]
        self.set_init_pos.pub_msg(start_pos)
        time.sleep(1)

        rclpy.spin_once(self.node)  # Spin the node to update the state
        self.state = [map_to_closest(num) for num in self.state]  # Map state to the closest grid resolution

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
        
        increment_count(self.state[0], self.state[1], self.state[2], action)

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
        while not state_changed(prev_state, self.state, action):
            if direction:
                self.move_command.pub_msg(self.state, direction)
            rclpy.spin_once(self.node)
            self.state = [map_to_closest(num) for num in self.state]


        #print("ROS State:", self.state)
        distance_to_pad = np.linalg.norm(self.state - self.landing_pad_position)

        self.state = [map_to_closest(num) for num in self.state]
            
        # Check if the drone is within the finite 3D space around the landing pad
        position = self.state[:3]

        if all(abs(p) < GRID_RESOL for p in position):
            self.landed = True
            done = True
        elif position[2] < GRID_RESOL:
            self.crashed = True
            done = True

        rclpy.spin_once(self.check_armed)

        #reset if crashed or after 3000 episodes
        if self.check_armed.crashed or self.eps > 3000:
            self.crashed = True

            gazebo_res_command = ["gz","service", "-s", "/world/iris_runway/control", "--reqtype", 
                                  "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
                                  "--timeout", "3000", "--req", "reset: {all: true}"]
            subprocess.Popen(gazebo_res_command)

            os.kill(self.ardupilot_process.pid, 2)
            os.kill(self.gazebo_process.pid, 2)
            #os.kill(self.mavros_process.pid, 2)

            time.sleep(3)

            self.launch_gazebo()
            self.launch_ardupilot()
            #self.launch_ros2_mavros()
            
            time.sleep(20)

            self.prepare_flight()
            self.eps = 0

        if self.step_counter > (6+6+8)/GRID_RESOL+10 or distance_to_pad > self.landing_pad_radius:
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
        
        return self.state, reward, done, done, self.landed
    
    
    def close(self):
        """
        Cleans up resources by killing the external processes and shutting down the ROS2 node.
        Writes state-action counter to file
        """
        # Write state-action counter to file
        with open("dict_S_a.txt", 'a') as file:
            for key_x in counter_dict:
                for key_y in counter_dict[key_x]:
                    for key_z in counter_dict[key_x][key_y]:
                        for key_a, count in counter_dict[key_x][key_y][key_z].items():
                            file.write(f"({key_x}, {key_y}, {key_z}, {key_a}) -> {count}\n")

        # Shutdown ROS2 and external processes
        self.node.destroy_node()
        os.kill(self.ardupilot_process.pid, 2)
        os.kill(self.gazebo_process.pid, 2)
        #os.kill(self.mavros_process, 2)
        rclpy.shutdown()
