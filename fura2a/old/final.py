#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from control import lqr, dlqr, ss
import math

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
        # self.m1 = 1.786614971490255
        self.m1 = 0.5334931362492104
        self.m2 = 0.08939794594871456
        self.l1 = 0.007967562168493848
        self.l2 = 0.06
        self.L1 = 0.075
        self.L2 = 0.12
        self.b1 = 0.0
        self.b2 = 0.0
        self.J0_hat = 0.0014758254169878584
        # self.J0_hat = 0.00283041
        self.J2_hat = 0.0004809817320359795
        
        self.theta1_weight = 0.0
        self.theta2_weight = 100.0
        self.dtheta1_weight = 10.0
        self.dtheta2_weight = 10.0
        self.u_weight = 0.01
        self.dt = 1/333

        self.A, self.B, self.Q, self.R = self.get_ABQR()
        self.K, self.S, self.E = self.get_KSE_d()

        print("A is", self.A)
        print("B is", self.B)
        print("Q is", self.Q)
        print("R is", self.R)
        
        print("K is", self.K)
        print("S is", self.S)
        print("E is", self.E)

        self.X = self.X = np.zeros((4, 1), dtype=float)

        self.u_max = 0.05
        self.position_threshold = 0.2

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
    
    def limit_minus_pi_pi(self, angle):
        angle = math.fmod(angle, 2 * math.pi)  # Wrap angle within [-2*pi, 2*pi]
        if angle > math.pi:
            angle -= 2 * math.pi  # Adjust if angle > pi
        elif angle < -math.pi:
            angle += 2 * math.pi  # Adjust if angle < -pi
        return angle
    
    # def swingup(self, position, velocity):
    #     theta = self.limit_minus_pi_pi(position - math.pi)
    #     E0 = self.m2 * self.g * self.l2
    #     KE = 0.5 * self.m2 * (self.l2**2) * (velocity**2) 
    #     PE = self.m2 * self.g * self.l2 * math.cos(theta)
    #     E = KE + PE
    #     d_E = E0 - E

    #     if d_E * velocity * math.cos(theta) > 0:
    #         return -float(self.u_max)  # Torque to increase energy
    #     else:
    #         return float(self.u_max)  # Torque in the opposite direction
        
    def swingup(self, position, velocity):
        """
        Smooth energy-based swing-up control. 
        """
        theta = self.limit_minus_pi_pi(position - math.pi)
        E0 = self.m2 * self.g * self.l2
        KE = 0.5 * self.m2 * (self.l2**2) * (velocity**2) 
        PE = self.m2 * self.g * self.l2 * math.cos(theta)
        E = KE + PE
        d_E = E0 - E

        # Scale the torque smoothly based on energy difference
        k = 0.1  # Scaling factor for smoothness
        torque = k * d_E * velocity * math.cos(theta)

        # Limit the torque to the maximum allowable range
        return np.clip(torque, -self.u_max, self.u_max)

        
    def do_lqr(self, X):
        X_ref =np.array([[0], [math.pi], [0], [0]])
        X[1] = self.limit_minus_pi_pi(X[1] - math.pi)
        u = (-self.K @ X).item()
        print(u)
        return u
        
    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            
            pendulum_position = msg.position[pendulum_index]
            # normalized_pendulum_position = self.limit_minus_pi_pi(pendulum_position)
            pendulum_velocity = msg.velocity[pendulum_index]

            self.X[0] = arm_position
            self.X[1] = pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            # print(self.limit_minus_pi_pi(pendulum_position))

            if math.pi - abs(self.limit_minus_pi_pi(pendulum_position)) <= self.position_threshold:
                self.effort_command =  self.do_lqr(self.X)
            else:
                self.effort_command = self.swingup(pendulum_position, pendulum_velocity)

            # print(self.swingup(pendulum_position, pendulum_velocity))
            # self.effort_command = self.swingup(pendulum_position, pendulum_velocity)

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
