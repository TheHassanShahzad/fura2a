#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from math import cos, pi
import numpy as np
from scipy.linalg import svd
from control import lqr, dlqr, ss

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
        # self.m1 = 0.07359375
        self.m1 = 0.3
        self.m2 = 0.105975
        self.l1 = 0.0375
        self.l2 = 0.0675
        self.sl2 = 0.061
        self.L1 = 0.08
        self.L2 = 0.135
        self.b1 = 0.0001
        self.b2 = 0.0003
        # self.J1 = 0.000141365966796875
        # self.J1 = 0.0005762689090000001
        # self.J2 = 0.00064749796875

        self.J1 = 1.42675159e-04
        self.J2 = 0.000162

        self.J2_hat = self.J2 + self.m2 * self.l2 * self.l2
        self.J0_hat = self.J1 + self.m1 * self.l1 * self.l1 + self.m2 * self.L1 * self.L1
        
        print("J2_hat is", self.J2_hat)
        print("J0_hat is", self.J0_hat)
        
        self.theta1_weight = 0.0
        self.theta2_weight = 100.0
        self.dtheta1_weight = 10.0
        self.dtheta2_weight = 30.0
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

        self.X = self.X = np.zeros((4, 1), dtype=float)
        self.kc = 0.075
        self.position_threshold = 0.15
        self.velocity_threshold = 0.2

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
        
    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    

    # def swing_up(self, pendulum_position, pendulum_velocity):
    #     E = ((self.m2 * (self.l2**2) * (pendulum_velocity**2)) / 2) - (self.m2 * self.g * self.l2 * (cos(pendulum_position)))
    #     E_desired = self.m2 * self.g * self.l2
    #     d_E = E - E_desired
    #     # print(d_E)
    #     u = self.kc * d_E * pendulum_velocity * cos(pendulum_position)
    #     return u

    def swing_up(self, pendulum_position, pendulum_velocity):
        E = ((self.m2 * (self.sl2**2) * (pendulum_velocity**2)) / 2) - (self.m2 * self.g * self.sl2 * (cos(pendulum_position)))
        E_desired = self.m2 * self.g * self.sl2
        d_E = E - E_desired
        # print(d_E)
        u = self.kc * d_E * pendulum_velocity * cos(pendulum_position)
        return u

    
    def do_lqr(self, X):
        X_ref = np.array([[0.0], [0.0], [0.0], [0.0]])
        u = self.K @ (X - X_ref)
        return u.item()


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

            # print (pendulum_position)

            self.X[0] = arm_position
            self.X[1] = pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            msg_to_publish = Float64MultiArray()

            if pi - abs(pendulum_position) <= self.position_threshold and abs(pendulum_velocity) <= self.velocity_threshold:
            # if pi - abs(pendulum_position) <= self.position_threshold:
                print("start lqr")
                print(self.do_lqr(self.X))  
                self.effort_command = self.do_lqr(self.X)
                # self.effort_command = self.lqr(self.K, self.X, self.u_max)
            else:
                print("no")
                self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
                
            # self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
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
