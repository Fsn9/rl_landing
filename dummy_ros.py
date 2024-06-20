import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped

class DummyROS(Node):

    def __init__(self):
        super().__init__('dummy_ros')
        self.publisher_ = self.create_publisher(TwistStamped, '/ap/cmd_vel', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.twist.linear.z = 1.0 if self.i % 2 == 0 else -1.0
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.twist)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    dummy = DummyROS()

    rclpy.spin(dummy)

    dummy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
