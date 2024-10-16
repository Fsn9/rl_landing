# rl_landing

It contains code that outputs navigational UAV commands. Specifically designed for autonomous landing and RL-based navigation. The commands are exchanged using a MAVLink API named `pymavlink.mavutil`.

The main scripts are:

`rl_landing/rl_landing/main.py` Is the main source code that connects to a remote UAV via mavlink;

`rl_landing/rl_landing/action_spaces.py` Contains the RL's action spaces;

`rl_landing/rl_landing/controller.py` Contains the navigational controller classes compatible with ROS2;

`rl_landing/rl_landing/dataset_loader.py` Contains a dataset loader that reads UAV remote pilotting data and creates a dataset object;

`rl_landing/rl_landing/networks.py` Contains several neural network architectures;

`rl_landing/rl_landing/replay_memory.py` Contains the replay memory RL class for off-policy methods;

---

The goal of this project is to implement a drone's autonomous landing model in a 3D environment using **DQN**. The drone is controlled through ROS 2 Humble, and the environment is visualized and managed using Gazebo with MAVROS and ArduPilot interfaces.

- **Custom ROS 2 environment**: The environment uses ROS 2 Humble for interacting with the drone in a simulated 3D space.
- **Gazebo and ArduPilot**: Simulated physics and drone dynamics are handled using Gazebo, while ArduPilot is used for flight control.
- **Observation space**: 3D coordinates \delta (X, Y, Z).
- **Action space**: 9 discrete actions like moving in different directions.

## **Requirements**

- **Python 3.10**
- **ROS2 Humble**
- **Gazebo Harmonic**
- **ArduPilot**
- **MAVROS**
- **Python Packages**:
  - `gym`
  - `h5py`
  - `torch`

### **Training**
A trained model will be saved as model in the current directory. Logs and Q-values for different states will be saved in a text file. A graph showing the return per episode will be generated and saved.
   ```bash
   python main_code.py train
   ```

### **Simulation Testing**
Requires ```model``` file generated during training. The agent will attempt to land the drone on the landing pad over several episodes. The script will display the rewards and episode durations.

```(x,y,z) -> ros2 topic: 'mavros/global_position/local'```

```actions -> ros2 topic: 'mavros/setpoint_position/local'```
   ```bash
   python main_code.py test
   ```

### **Real Testing**
Requires ```model``` file generated during training. The agent will attempt to land the drone on the landing pad over several episodes. The script will display the rewards and episode durations.

``` (x,y) -> ros2 topic: 'mavros/global_position/local' ```

``` (z)   -> ros2 topic: 'mavros/rangefinder/rangefinder' ```

```actions -> ros2 topic: 'mavros/setpoint_velocity/cmd_vel'```


   ```bash
   python test_real.py
   ```

