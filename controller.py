import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped

"""
class ROS2Controller that has:
    a publisher to /ap/cmd_vel

    a subscriber to /ap/pose/filtered
    a subscriber to /ap/gps_origin_filtered
"""
class ROS2Controller(Node):

    def __init__(self, name='ros2 controller'):
        super().__init__(name)
        self.publisher_ = self.create_publisher(TwistStamped, '/ap/cmd_vel', 10)
        self.i = 0

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.twist)
        self.i += 1
    
    def __call__(self):
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
        
        self.i += 1

        self.publisher_.publish(msg)

        self.get_logger().info('Publishing: "%s"' % msg.twist)

"""
DQN method
"""
class DQN(ROS2Controller):
    def __init__(self):
        super().__init__('dqn')

"""
Dummy method
"""
class Dummy(ROS2Controller):
    def __init__(self):
        ROS2Controller.__init__('dummy')

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
