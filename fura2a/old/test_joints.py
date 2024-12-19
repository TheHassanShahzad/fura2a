import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        # Publisher
        self.publisher = self.create_publisher(JointState, '/topic_based_joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Add subscriber to monitor joint_states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['arm_joint', 'pendulum_joint']
        msg.position = [1.57, 0.5]
        msg.velocity = [0.1, -0.2]
        msg.effort = [0.0, 0.0]
        self.publisher.publish(msg)
        self.get_logger().info('Published to topic_based_joint_states')

    def joint_states_callback(self, msg):
        self.get_logger().info(f'Received on joint_states - stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

def main():
    rclpy.init()
    node = JointStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()