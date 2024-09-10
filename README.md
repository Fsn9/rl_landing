# rl_landing

It contains code that outputs navigational UAV commands. Specifically designed for autonomous landing and RL-based navigation. The commands are exchanged using a MAVLink API named `pymavlink.mavutil`.

The main scripts are:

`rl_landing/rl_landing/main.py` Is the main source code that connects to a remote UAV via mavlink;

`rl_landing/rl_landing/action_spaces.py` Contains the RL's action spaces;

`rl_landing/rl_landing/controller.py` Contains the navigational controller classes compatible with ROS2;

`rl_landing/rl_landing/dataset_loader.py` Contains a dataset loader that reads UAV remote pilotting data and creates a dataset object;

`rl_landing/rl_landing/networks.py` Contains several neural network architectures;

`rl_landing/rl_landing/replay_memory.py` Contains the replay memory RL class for off-policy methods;
