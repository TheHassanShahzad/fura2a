#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from math import cos, pi


class JointStateProcessor(Node):
    def __init__(self):
        super().__init__('joint_state_processor')

        # ROS2 subscriber to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)

        # ROS2 publisher for control efforts
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            'arm_cont/commands',
            10)

        # Physical parameters
        self.m2 = 0.0893979  # Pendulum mass
        self.l2 = 0.06       # Pendulum COM length
        self.L1 = 0.075      # Arm length
        self.g = 9.81        # Gravity
        self.J0 = 0.012765   # Arm moment of inertia
        self.J2 = 0.000479   # Pendulum moment of inertia
        self.kc = 1.5        # Swing-up gain (adjusted for more energy)
        self.Ts = 0.01       # Sampling time (increased for simulation stability)

        # Linearized matrices at upright position
        self.compute_linearized_matrices()

        self.Ac = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [self.A31, self.A32, 0, 0],
            [self.A41, self.A42, 0, 0]
        ])

        self.Bc = np.array([
            [0],
            [0],
            [self.B31],
            [self.B41]
        ])

        # Discretize the system
        self.Ad, self.Bd, _, _, _ = cont2discrete((self.Ac, self.Bc, None, None), self.Ts)

        # LQR setup
        self.Q = np.diag([10, 100, 1, 10])  # State cost
        self.R = np.array([[0.1]])            # Input cost
        self.K = self.lqr(self.Ad, self.Bd, self.Q, self.R)

        # State vector
        self.X = np.zeros((4, 1))

        # Thresholds for switching between swing-up and LQR
        self.angle_threshold = 0.25       # ~8.6 degrees
        self.velocity_threshold = 0.3     # Threshold for angular velocity

        self.effort_command = 0.0
        self.u_max = 1.0  # Max allowable effort

        self.get_logger().info('JointStateProcessor initialized')

    def compute_linearized_matrices(self):
        """
        Compute the A and B matrices linearized about the upright position.
        """
        denominator = (self.J0 * self.J2 - self.m2**2 * self.L1**2 * self.l2**2)

        self.A31 = (self.g * self.m2**2 * self.l2**2 * self.L1) / denominator
        self.A32 = -(self.g * self.m2 * self.l2 * self.J0) / denominator
        self.A41 = -(self.g * self.m2 * self.l2 * self.J0) / denominator
        self.A42 = (self.g * self.m2**2 * self.l2**2 * self.L1) / denominator

        self.B31 = self.J2 / denominator
        self.B41 = (self.m2 * self.L1 * self.l2) / denominator

    def lqr(self, A, B, Q, R):
        """
        Solve the Discrete-time Algebraic Riccati equation and compute LQR gain.
        """
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

    def swing_up(self, theta, theta_dot):
        """
        Energy-based swing-up control.
        """
        E = 0.5 * self.m2 * (self.l2**2) * (theta_dot**2) - self.m2 * self.g * self.l2 * cos(theta)
        E_desired = self.m2 * self.g * self.l2
        d_E = E - E_desired
        u = self.kc * d_E * theta_dot * cos(theta)
        return np.clip(u, -self.u_max, self.u_max)

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        """
        return (angle + pi) % (2 * pi) - pi

    def listener_callback(self, msg):
        """
        Callback function to process joint states and publish control efforts.
        """
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')

            # Extract positions and velocities
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            pendulum_position = self.normalize_angle(msg.position[pendulum_index])
            pendulum_velocity = msg.velocity[pendulum_index]

            # Update state vector
            self.X[0] = arm_position
            self.X[1] = pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            # Determine control mode
            angle_error = abs(self.normalize_angle(pendulum_position - pi))
            if angle_error < self.angle_threshold and abs(pendulum_velocity) < self.velocity_threshold:
                # Stabilize using LQR
                self.effort_command = -self.K @ self.X
                self.effort_command = np.clip(self.effort_command, -self.u_max, self.u_max)
                self.get_logger().info(f'LQR Control: {self.effort_command[0]}')
            else:
                # Swing up control
                self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
                self.get_logger().info(f'Swing-Up Control: {self.effort_command}')

            # Publish effort command
            msg_to_publish = Float64MultiArray()
            msg_to_publish.data = [float(self.effort_command)]
            self.publisher_.publish(msg_to_publish)

        except ValueError as e:
            self.get_logger().warn(f'Error processing joint states: {e}')


def main(args=None):
    rclpy.init(args=args)
    joint_state_processor = JointStateProcessor()
    rclpy.spin(joint_state_processor)
    joint_state_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
