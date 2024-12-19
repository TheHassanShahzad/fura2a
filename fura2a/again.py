#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from scipy.linalg import svd
import numpy as np
from control import lqr, dlqr, ss
import math, csv

class JointStateProcessor(Node):
    def __init__(self):
        super().__init__('joint_state_processor')
        
        # Create subscriber to /joint_states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        # self.subscription  # prevent unused variable warning
        
        # Create publisher to "arm_cont/command" topic
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            'arm_cont/commands',
            10)
        
        self.g = 9.80665
        self.m1 = 0.07359375
        self.m2 = 0.105975
        self.l1 = 0.0375
        self.l2 = 0.0675
        self.L1 = 0.08
        self.L2 = 0.135
        self.b1 = 0.0001
        self.b2 = 0.0003
        self.J1 = 0.000141365966796875
        self.J2 = 0.00064749796875

        self.J2_hat = self.J2 + self.m2 * self.l2 * self.l2
        self.J0_hat = self.J1 + self.m1 * self.l1 * self.l1 + self.m2 * self.L1 * self.L1
        
        print("J2_hat is", self.J2_hat)
        print("J0_hat is", self.J0_hat)
        
        self.theta1_weight = 0.0
        self.theta2_weight = 100.0
        self.dtheta1_weight = 10.0
        self.dtheta2_weight = 10.0
        self.u_weight = 0.01
        self.dt = 1/333

        self.A, self.B, self.Q, self.R = self.get_ABQR()
        self.is_controllable()
        self.K, self.S, self.E = self.get_KSE_d()

        print("A is", self.A)
        print("B is", self.B)
        print("Q is", self.Q)
        print("R is", self.R)
        
        print("K is", self.K)
        print("S is", self.S)
        print("E is", self.E)

        self.count = 0
        self.final_count = 330*10
        self.data  = []

    def is_controllable(self):
        n = self.A.shape[0]

        C = self.B
        for i in range(1, n):
            C = np.hstack((C, np.linalg.matrix_power(self.A, i).dot(self.B)))

        # Compute the rank
        rank = np.linalg.matrix_rank(C)

        # Check if the system is controllable
        if rank == n:
            print("The system is controllable.")
        else:
            print("The system is NOT controllable.")
    
    def get_ABQR(self):
        denominator = self.J0_hat * self.J2_hat - (self.m2**2.0) * (self.L1**2.0) * (self.l2**2.0)

        A32 = (self.g * (self.m2**2.0) * (self.l2**2.0) * self.L1) / denominator
        A33 = (-self.b1 * self.J2_hat) / denominator
        A34 = (-self.b2 * self.m2 * self.l2 * self.L1) / denominator

        A42 = (self.g * self.m2 * self.l2 * self.J0_hat) / denominator
        A43 = (-self.b1 * self.m2 * self.l2 * self.L1) / denominator
        A44 = (-self.b2 * self.J0_hat) / denominator

        B31 = (self.J2_hat) / denominator
        B41 = (self.m2 * self.L1 * self.l2) / denominator
        B32 = (self.m2 * self.L1 * self.l2) / denominator
        B42 = (self.J0_hat) / denominator

        A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, A32, A33, A34], [0.0, A42, A43, A44]])
        B = np.array([[0.0], [0.0], [B31], [B41]])

        Q = np.array(
            [
                [self.theta1_weight, 0.0, 0.0, 0.0],
                [0.0, self.theta2_weight, 0.0, 0.0],
                [0.0, 0.0, self.dtheta1_weight, 0.0],
                [0.0, 0.0, 0.0, self.dtheta2_weight],
            ]
        )
        R = np.array([self.u_weight])

        return A,B,Q,R

    def get_KSE_d(self):
        C = np.identity(4)
        # D = np.zeros(4).transpose()
        D = np.array([[0.0], [0.0], [0.0], [0.0]])

        dsys = ss(self.A, self.B, C, D, self.dt)
        K, S, E = dlqr(dsys, self.Q, self.R)

        return K, S, E
        
    def normalise_minus_pi_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def swing_up(self, pendulum_position, pendulum_velocity):
        """
        A basic energy-based swing-up controller.
        The goal is to bring the pendulum from hanging down (0 radians)
        to the upright position (pi radians) by gradually increasing its energy.
        
        Parameters
        ----------
        pendulum_position : float
            Current pendulum angle (in radians), normalized to [-pi, pi]
            where 0 is the downward position and ±pi is the upright position.
        pendulum_velocity : float
            Current angular velocity of the pendulum (in radians per second).
            
        Returns
        -------
        float
            Control torque to apply at the joint.
        """

        # Extract necessary parameters
        m = self.m2      # Mass of the pendulum bob
        g = self.g        # Gravitational acceleration
        l = self.l2       # Distance from the pivot to the pendulum mass center
        J = self.J2_hat   # Effective moment of inertia about the pivot

        # Desired energy for the pendulum to be at the top (pi)
        # At the bottom (0 rad), potential energy is at a minimum.
        # To raise it to pi (upwards), the potential energy difference is 2*m*g*l.
        E_des = 2 * m * g * l

        # Current total energy of the pendulum:
        # Kinetic energy: (1/2)*J*ω²
        # Potential energy: m*g*l*(1 - cos(theta))
        # At theta=0 (down), PE = m*g*l*(1 - cos(0)) = m*g*l*(1 - 1) = 0
        # At theta=pi (up), PE = m*g*l*(1 - cos(pi)) = m*g*l*(1 - (-1)) = 2*m*g*l
        E = 0.5 * J * (pendulum_velocity**2) + m*g*l*(1 - np.cos(pendulum_position))

        # Energy difference
        E_diff = E - E_des

        # Gain for the controller
        # If E_diff < 0 (less than desired energy), we need to add energy.
        # If E_diff > 0 (more than desired), we might want to stop adding energy.
        # A common approach is to apply torque proportional to the product of angular velocity
        # and energy difference, encouraging energy to build up as the pendulum passes through bottom.
        k = 0.2  # Tune this gain as necessary
        
        # Control law: apply torque based on energy difference and velocity.
        # The sign of the torque is chosen so that if we have less energy than needed,
        # we push in the direction that will increase the pendulum’s amplitude.
        u = k * pendulum_velocity * E_diff

        return u

        
    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            pendulum_position = msg.position[pendulum_index]
            pendulum_velocity = msg.velocity[pendulum_index]

            normalized_pendulum_position = self.normalise_minus_pi_pi(pendulum_position)
            # print("not normalized pendulum position is", pendulum_position, "normalized pendulum position is", normalized_pendulum_position)

            self.effort_command = self.swing_up(normalized_pendulum_position, pendulum_velocity)
            # print(" effort command is", self.effort_command)

            print(normalized_pendulum_position)
            msg_to_publish = Float64MultiArray()
            msg_to_publish.data = [self.effort_command]
            self.publisher_.publish(msg_to_publish)

            if self.count == self.final_count:
                with open("normalised_pendulum_positions].csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    # Write the list of floats as a single row
                    writer.writerow(self.data)

            print(self.count)
            self.data.append(normalized_pendulum_position)
            self.count += 1

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
