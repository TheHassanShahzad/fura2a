import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class PendulumDataRecorder(Node):
    def __init__(self):
        super().__init__('pendulum_data_recorder')
        
        # Initialize the list to store pendulum_joint position data
        self.pendulum_positions = []
        self.data_limit = 20  # Number of data points before writing to file
        
        # Subscriber to the /joint_states topic
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )
        
        self.get_logger().info('Pendulum Data Recorder Node has been started.')

    def listener_callback(self, msg):
        try:
            # Extract pendulum_joint position
            pendulum_index = msg.name.index('pendulum_joint')
            pendulum_position = msg.position[pendulum_index]
            
            # Append the position to the list
            self.pendulum_positions.append(pendulum_position)
            
            # Check if the list has reached the data limit
            if len(self.pendulum_positions) >= self.data_limit:
                self.write_to_file()
                self.pendulum_positions = []  # Reset the list after writing
                
        except ValueError as e:
            self.get_logger().error(f'Joint name not found: {e}')
    
    def write_to_file(self):
        try:
            with open('pendulum_positions.txt', 'a') as file:
                for position in self.pendulum_positions:
                    file.write(f"{position}\n")
            self.get_logger().info('Pendulum positions written to file.')
        except Exception as e:
            self.get_logger().error(f'Error writing to file: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PendulumDataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
