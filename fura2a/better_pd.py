#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from math import cos, pi
import numpy as np

class JointStateProcessor(Node):
    def __init__(self):
        super().__init__('joint_state_processor')
        
        # Create subscriber to /joint_states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)

        # Create publisher to "arm_cont/command" topic
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            'arm_cont/commands',
            10)
        
        # System parameters
        self.g = 9.80665
        self.m1 = 0.07359375
        self.m2 = 0.105975
        self.l1 = 0.0375
        self.l2 = 0.0675
        self.sl2 = 0.061
        self.L1 = 0.08
        self.L2 = 0.135
        self.b1 = 0.0001
        self.b2 = 0.0003

        # State vector: [arm_position, pendulum_position, arm_velocity, pendulum_velocity]
        self.X = np.zeros((4, 1), dtype=float)

        # Swing-up gain
        self.kc = 0.075

        # Thresholds for switching from swing-up to PD control
        self.position_threshold = 0.4
        self.velocity_threshold = 0.05

        # PD Controller Gains
        # These are initial guesses and will need to be tuned
        # Kp influences how strongly the controller acts based on position error
        # Kd influences how strongly the controller acts based on velocity error
        self.Kp = 0.7
        self.Kd = 0.1

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def swing_up(self, pendulum_position, pendulum_velocity):
        # Energy-based swing-up control
        E = ((self.m2 * (self.sl2**2) * (pendulum_velocity**2)) / 2) - (self.m2 * self.g * self.sl2 * (cos(pendulum_position)))
        E_desired = self.m2 * self.g * self.sl2
        d_E = E - E_desired
        u = self.kc * d_E * pendulum_velocity * cos(pendulum_position)
        return u

    def pd_control(self, pendulum_position, pendulum_velocity, arm_position, arm_velocity):
        # Desired pendulum position
        desired_pendulum_position = pi

        # Desired arm position (you can set this to 0 or some other neutral value)
        desired_arm_position = 0.0

        # Pendulum error
        pendulum_error = self.normalize_angle(desired_pendulum_position - pendulum_position)
        pendulum_velocity_error = 0.0 - pendulum_velocity

        # Arm error
        arm_error = desired_arm_position - arm_position
        arm_velocity_error = 0.0 - arm_velocity

        # PD control law
        pendulum_control_effort = self.Kp * pendulum_error + self.Kd * pendulum_velocity_error
        arm_control_effort = 0.1 * (self.Kp * arm_error + self.Kd * arm_velocity_error)  # Scale arm control

        # Combine efforts
        control_effort = pendulum_control_effort + arm_control_effort
        return control_effort



    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            
            pendulum_position = msg.position[pendulum_index]
            pendulum_position = self.normalize_angle(pendulum_position)
            pendulum_velocity = msg.velocity[pendulum_index]

            self.X[0] = arm_position
            self.X[1] = pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            msg_to_publish = Float64MultiArray()

            # Check if pendulum is close to inverted position
            if pi - abs(pendulum_position) <= self.position_threshold:
                # Use PD controller to stabilize at the top
                print("Using PD controller")
                self.effort_command = self.pd_control(pendulum_position, pendulum_velocity, arm_position, arm_velocity)
            else:
                # Use swing-up controller
                print("Using swing-up controller")
                self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
            
            msg_to_publish.data = [self.effort_command]
            
            # Publish the message
            self.publisher_.publish(msg_to_publish)

        except ValueError:
            # Handle the case where the joint name is not found
            self.get_logger().warn('Joint name not found in JointState message')
            

def main(args=None):
    rclpy.init(args=args)
    joint_state_processor = JointStateProcessor()
    rclpy.spin(joint_state_processor)
    joint_state_processor.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
