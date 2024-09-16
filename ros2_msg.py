import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs import srv
from mavros_msgs.msg import State

import time

qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


GRID_RESOL = 0.25


class GoToXYZ_vel(Node):
    """
    A ROS 2 node for controlling movement via velocity commands.

    Features:
    - Publishes to the topic `mavros/setpoint_velocity/cmd_vel`.

    Methods:
    - `pub_msg(self, actual, dir)`: Publishes velocity commands based on the current 
      position (`actual`) and desired movement direction (`dir`).
    """
    
    def __init__(self):
        super().__init__('move_command_vel')
        self.publisher_ = self.create_publisher(TwistStamped, 'mavros/setpoint_velocity/cmd_vel', 10)
       

    def pub_msg(self, actual, dir):
        mesg = TwistStamped()

        vel_test = 0.25
        slow_vel = 0.15

        # Mapping directions to linear velocity changes
        velocity_vectors = {
            'left'    : (-vel_test,  0.0,       0.0),  
            'right'   : (+vel_test,  0.0,       0.0),
            'backward': ( 0.0,      -vel_test,  0.0),
            'forward' : ( 0.0,      +vel_test,  0.0),
           #'down'    : ( 0.0,      0.0,        -vel_test or -slow_vel ),
            'lb'      : (-vel_test, -vel_test,  0.0),
            'lf'      : (-vel_test, +vel_test,  0.0),
            'rb'      : (+vel_test, -vel_test,  0.0),
            'rf'      : (+vel_test, +vel_test,  0.0)
        }

        # Handle directions with z-axis velocity or more complex conditions
        if dir == 'down':
            if actual[2] < 2.0:                  # Slow down when below certain height
                mesg.twist.linear.z = -slow_vel  
            else:
                mesg.twist.linear.z = -vel_test  # Normal descent

        # Apply velocities for other directions
        elif dir in velocity_vectors:
            delta_x, delta_y, delta_z = velocity_vectors[dir]
            mesg.twist.linear.x = delta_x
            mesg.twist.linear.y = delta_y
            mesg.twist.linear.z = delta_z  # z is 0.0 for these directions
        else:
            raise ValueError(f"Invalid direction: {dir}")

        # Publish the message to the topic
        self.publisher_.publish(mesg)


class GoToXYZ(Node):
    """
    A ROS 2 node for controlling movement by setting positions.
    
    Features:
    - Publishes position commands to the topic `mavros/setpoint_position/local`.

    Methods:
    - `pub_msg(self, actual, dir)`: Publishes a position message based on the current 
      position (`actual`) and the desired direction (`dir`).
    """
    
    def __init__(self):
        super().__init__('move_command_pos')
        self.publisher_ = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
       

    def pub_msg(self, actual, dir):
        mesg = PoseStamped()

        # Mapping directions to corresponding movement vectors
        direction_vectors = {
        'left'    : (-GRID_RESOL,  0.0,         0.0),
        'right'   : (+GRID_RESOL,  0.0,         0.0),
        'forward' : ( 0.0,        +GRID_RESOL,  0.0),
        'backward': ( 0.0,        -GRID_RESOL,  0.0),
        'down'    : ( 0.0,         0.0,        -GRID_RESOL),
        'lb'      : (-GRID_RESOL, -GRID_RESOL,  0.0),
        'lf'      : (-GRID_RESOL, +GRID_RESOL,  0.0),
        'rb'      : (+GRID_RESOL, -GRID_RESOL,  0.0),
        'rf'      : (+GRID_RESOL, +GRID_RESOL,  0.0)
        }

        # Update position based on direction
        if dir in direction_vectors:
            delta_x, delta_y, delta_z = direction_vectors[dir]
            mesg.pose.position.x = actual[0] + delta_x
            mesg.pose.position.y = actual[1] + delta_y
            mesg.pose.position.z = actual[2] + delta_z
        else:
            raise ValueError(f"Invalid direction: {dir}")

        # Set default orientation
        mesg.pose.orientation.x = 0.0
        mesg.pose.orientation.y = 0.0
        mesg.pose.orientation.z = 0.0
        mesg.pose.orientation.w = 1.0

        # Publish the message to the topic
        self.publisher_.publish(mesg)


class SetModeGuided(Node):
    """
    A ROS 2 node to set the drone's mode to 'GUIDED' using a service call.

    This class handles the transition of the MAVROS flight controller mode 
    to 'GUIDED' by sending an asynchronous service request to `/mavros/set_mode`.
    
    """

    def __init__(self):
        super().__init__('set_mode_guided')
        
        # Create a service client for the MAVROS 'SetMode' service
        self.service_client = self.create_client(srv.SetMode, '/mavros/set_mode')
        
        # Wait until the service is available, logging if not immediately ready
        while not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Initialize the service request object
        self.req = srv.SetMode.Request()
    
    def send_request(self):
        """Sends a request to switch the mode to 'GUIDED'."""
        
        # Set the desired flight mode to 'GUIDED'
        self.req.custom_mode = 'GUIDED'
        
        # Call the service asynchronously
        self.future = self.service_client.call_async(self.req)
        
        # Spin the node until the future is complete (blocking)
        rclpy.spin_until_future_complete(self, self.future)
        
        # Check if the future completed successfully, and log the result
        if self.future.result() is not None:
            self.get_logger().info(f"Mode changed successfully: {self.future.result()}")
        else:
            self.get_logger().error(f"Failed to change mode: {self.future.exception()}")




class ArmThrottle(Node):
    """
    A ROS 2 node to arm the throttle using a service call to MAVROS.

    This class sends a request to the `/mavros/cmd/arming` service to arm the 
    drone's throttle. If the request fails, it retries at regular intervals 
    until successful.
    """

    def __init__(self):
        super().__init__('arm_throttle')
        
        # Create the service client for the arming service
        self.service_client = self.create_client(srv.CommandBool, '/mavros/cmd/arming')
        
        # Wait for the arming service to become available
        while not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the arming service to become available...')

        # Initialize the service request object
        self.req = srv.CommandBool.Request()

    def send_request(self):
        """Send the arming request and retry until successful."""
        
        self.req.value = True  # Set to arm the throttle
        attempt = 0
        
        while True:
            attempt += 1
            # Send the service request asynchronously
            self.future = self.service_client.call_async(self.req)
            
            # Wait for the result to complete
            rclpy.spin_until_future_complete(self, self.future)
            
            # Check if the request succeeded
            if self.future.result().success:
                self.get_logger().info(f'Throttle armed successfully on attempt {attempt}.')
                break
            else:
                self.get_logger().error(f'Arming failed on attempt {attempt}. Retrying in 5 seconds...')
                time.sleep(5)  # Wait before retrying

            
class Takeoff(Node):
    """
    A ROS 2 node to initiate a takeoff using MAVROS.

    This class sends a request to the `/mavros/cmd/takeoff` service to take off 
    the drone to a specified altitude. If the request fails, it retries until successful.
    """

    def __init__(self):
        super().__init__('takeoff')
        
        # Create the service client for the takeoff service
        self.service_client = self.create_client(srv.CommandTOL, '/mavros/cmd/takeoff')
        
        # Wait for the takeoff service to become available
        while not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the takeoff service to become available...')

        # Initialize the service request object
        self.req = srv.CommandTOL.Request()

    def send_request(self, altitude=3.0):
        """Send the takeoff request to reach the specified altitude and retry if necessary."""
        
        self.req.altitude = altitude  # Set the desired altitude for takeoff
        attempt = 0
        
        while True:
            attempt += 1
            # Send the takeoff service request asynchronously
            self.future2 = self.service_client.call_async(self.req)
            
            # Wait for the result to complete
            rclpy.spin_until_future_complete(self, self.future2)
            
            # Check if the takeoff request succeeded
            if self.future2.result().success:
                self.get_logger().info(f'Takeoff successful on attempt {attempt} to altitude {altitude}.')
                break
            else:
                self.get_logger().error(f'Takeoff failed on attempt {attempt}. Retrying in 5 seconds...')
                time.sleep(5)  # Wait before retrying


class SetPosition(Node):
    """
    A ROS 2 node for publishing position commands to the drone.

    This class sends `PoseStamped` messages to the `/mavros/setpoint_position/local` 
    topic to set the desired position of the drone. It updates the position 
    with provided coordinates and sets a default orientation.
    """

    def __init__(self):
        super().__init__('move_command')
        
        # Create a publisher for sending PoseStamped messages to the position setpoint topic
        self.publisher_ = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)

    def pub_msg(self, pos):
        """
        Publish a position message to set the desired position of the drone.
        
        Parameters:
        - pos (tuple): A tuple of (x, y, z) coordinates representing the desired position.
        """
        
        # Create a PoseStamped message
        mesg = PoseStamped()
        
        # Set the position based on provided coordinates
        mesg.pose.position.x = pos[0]
        mesg.pose.position.y = pos[1]
        mesg.pose.position.z = pos[2]
        
        # Set default orientation
        mesg.pose.orientation.x = 0.0
        mesg.pose.orientation.y = 0.0
        mesg.pose.orientation.z = 0.0
        mesg.pose.orientation.w = 1.0

        # Publish the message to the topic
        self.publisher_.publish(mesg)


class CheckArmed(Node):
    """
    A ROS 2 node that monitors the armed status of a drone or robot.

    This class subscribes to the `/mavros/state` topic to receive updates on the 
    vehicle's state and checks if the vehicle is armed. If the vehicle is not armed, 
    it sets the `crashed` attribute to `True`.
    """

    def __init__(self):
        super().__init__('state_subscriber')
        
        # Initialize the crashed attribute
        self.crashed = False
        
        # Create a subscription to the MAVROS state topic
        self.subscription = self.create_subscription(State, '/mavros/state', self.check_armed, 10)

    def check_armed(self, msg):
        """
        Callback function to process incoming state messages and update the crashed status.
        
        Parameters:
        - msg (State): The state message received from the `/mavros/state` topic.
        """
        
        # Update the crashed attribute based on the armed status
        if not msg.armed:
            self.crashed = True
            self.get_logger().warn('Vehicle is not armed. Possible crash detected.')


