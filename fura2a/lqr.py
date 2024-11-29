#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from scipy.signal import cont2discrete

from math import cos, pi
import numpy as np
from scipy.linalg import solve_discrete_are

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
        
        self.mp = 0.0893979 
        self.ma = 1.7866149
        self.lp = 0.06
        self.g = 9.81
        # self.ia = 0.002830449 
        # self.ip = 0.000358709 
        self.ia = 0.00283041
        self.ip = 0.000322832
        self.kc = 0.2

        alpha = self.ip + self.mp * self.lp**2
        beta = self.lp * self.mp * self.lp
        delta = self.ia + self.mp * self.lp**2

        # Denominator for A and B matrices
        denominator = alpha * delta - beta**2
        
        self.Ts = 0.01

        self.Ac = np.array([
            [0, 1, 0, 0],
            [0, 0, beta * self.g * self.lp * self.mp / denominator, 0],
            [0, 0, 0, 1],
            [0, 0, -delta * self.g * self.lp * self.mp / denominator, 0]
        ])

        # B matrix
        self.Bc = np.array([
            [0],
            [1 / denominator],
            [0],
            [-beta / denominator]
        ])

        self.Cc = np.array([[1,0,0,0], [0,0,1,0]])
        self.Dc = 0

        self.Ad, self.Bd, self.Cd, self.Dd, _ = self.discretize(self.Ac, self.Bc, self.Cc, self.Dc, self.Ts)

        self.Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.R = np.array([[1]])

        self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ (self.Bd.T @ self.P @ self.Ad)
        
        self.X = np.array([[0],[0],[0],[0]])

        self.error_threshold = 0.08
        self.effort_command = 0
        
    def discretize(self, Ac, Bc, Cc, Dc, Ts):
        return cont2discrete((Ac, Bc, np.eye(Ac.shape[0]), 0), Ts)
    
    def swing_up(self, mp, lp, theta, theta_dot, g, kc):
        E = ((mp*(lp**2)*(theta_dot**2))/2) - (mp*g*lp*(cos(theta)))
        E_desired = mp*g*lp
        d_E = E-E_desired
        u = kc*d_E*theta_dot*cos(theta)
        return u
    
    def lqr(self, K, x):
        u = -K*x
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

            self.X[0] = arm_position
            self.X[1] = arm_velocity
            self.X[2] = pendulum_position
            self.X[3] = pendulum_velocity

            msg_to_publish = Float64MultiArray()

            self.effort_command = self.swing_up(self.mp, self.lp, pendulum_position, pendulum_velocity, self.g, self.kc)

            # error = abs(pendulum_position - pi)
            # print("error")
            # if error >= self.error_threshold:
            #     # print("swing")
            #     self.effort_command = self.swing_up(self.mp, self.lp, pendulum_position, pendulum_velocity, self.g, self.kc)
            # else:
            #     print("lqr!!!!!!!!!!!!!!!!")
            #     self.effort_command = self.lqr(self.K, self.X)
            
            msg_to_publish.data = [self.effort_command]
            
            # Publish the message
            self.publisher_.publish(msg_to_publish)
            # self.get_logger().info(f'Published value: {self.effort_command}')
            
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
