#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from control import lqr, dlqr, ss
from math import cos

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
        self.m1 = 1.786614971490255
        self.m2 = 0.08939794594871456
        self.l1 = 0.007967562168493848
        self.l2 = 0.06
        self.L1 = 0.075
        self.L2 = 0.12
        self.b1 = 0.0
        self.b2 = 0.0
        self.J0_hat = 0.00283041
        self.J2_hat = 0.000322832
        
        self.theta1_weight = 0.0
        self.theta2_weight = 10.0
        self.dtheta1_weight = 1.0
        self.dtheta2_weight = 5.0
        self.u_weight = 0.2
        self.dt = 1/330
        self.kc = 7

        self.A, self.B, self.Q, self.R = self.get_ABQR()
        self.K, self.S, self.E = self.get_KSE_d()

        print(self.K)
        print(self.K.shape)
        self.X = self.X = np.zeros((4, 1), dtype=float)

        self.position_threshold = 0.25
        self.velocity_threshold = 0.3

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
        return (angle + 2 * np.pi) % (2 * np.pi)
    

    def swing_up(self, theta, theta_dot):
        E = ((self.m2 * (self.l2**2) * (theta_dot**2)) / 2) - (self.m2 * self.g * self.l2 * (cos(theta)))
        E_desired = self.m2 * self.g * self.l2
        d_E = E - E_desired
        print(d_E)
        u = self.kc * d_E * theta_dot * cos(theta)
        # print(u)
        return u

    # def swing_up(self, theta, theta_dot):
    #     E = ((self.m2 * (self.l2**2) * (theta_dot**2)) / 2) - (self.m2 * self.g * self.l2 * (cos(theta)))
    #     E_desired = self.m2 * self.g * self.l2
    #     d_E = E - E_desired

    #     # Add damping when near the top position
    #     damping = 1 * theta_dot if abs(theta - np.pi) < self.position_threshold else 0.0  # Tune threshold and damping factor
    #     u = self.kc * d_E * theta_dot * cos(theta) - damping
    #     return u


    # def swing_up(self, theta, theta_dot):
    #     """
    #     Swing-Up Controller.

    #     Parameters:
    #     - theta: Pendulum angle (rad), 0 corresponds to downward position
    #     - theta_dot: Angular velocity (rad/s)

    #     Returns:
    #     - u: Control input (torque) for swing-up
    #     """
    #     # Total energy of the pendulum
    #     E = ((self.m2 * (self.l2**2) * (theta_dot**2)) / 2) - (self.m2 * self.g * self.l2 * np.cos(theta))

    #     # Desired energy at the upright position
    #     E_desired = self.m2 * self.g * self.l2

    #     # Energy difference
    #     d_E = E - E_desired

    #     umax = 1
    #     # Control input based on energy error
    #     if d_E * theta_dot * np.cos(theta) < 0:
    #         u = umax  # Add energy to the pendulum
    #     else:
    #         u = -umax  # Remove energy from the pendulum

    #     # Apply a proportional control gain
    #     u = self.kc * u

    #     # Limit the control input to the maximum torque
    #     u = np.clip(u, -umax, umax)

    #     return -u
    
    def new_swingup(self, theta, theta_dot):
        alpha = 2
        E = ((self.m2 * (self.l2**2) * (theta_dot**2)) / 2) - (self.m2 * self.g * self.l2 * (cos(theta)))
        E_desired = self.m2 * self.g * self.l2
        d_E = E - E_desired
        print(d_E)
        u = self.kc * np.sign(theta_dot) * (abs(d_E)**alpha)
        return -u

    def do_lqr(self, X):
        X_ref = np.array([[0], [np.pi], [0], [0]])
        u = -self.K@(X - X_ref)
        # print(u.item())
        # print(u.shape)
        return u.item()
    
    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            
            pendulum_position = msg.position[pendulum_index]
            normalized_pendulum_position = self.normalize_angle(pendulum_position)
            pendulum_velocity = msg.velocity[pendulum_index]

            self.X[0] = arm_position
            self.X[1] = normalized_pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            # if abs(normalized_pendulum_position - np.pi) <= self.position_threshold and pendulum_velocity <= self.velocity_threshold:
            # if abs(normalized_pendulum_position - np.pi) <= self.position_threshold:
            #     # print("do lqr")
            #     self.effort_command = self.do_lqr(self.X)
            # else:
            #     # print("continue swingup")
            #     # self.effort_command = self.swing_up(normalized_pendulum_position, pendulum_velocity)
            #     self.effort_command = self.new_swingup(normalized_pendulum_position, pendulum_velocity)

            self.effort_command = self.new_swingup(normalized_pendulum_position, pendulum_velocity)
            # print(self.X[3 ])
            msg_to_publish = Float64MultiArray()
            msg_to_publish.data = [self.effort_command]
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
