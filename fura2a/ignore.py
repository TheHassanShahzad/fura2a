#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete

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
        
        self.mp = 0.0893979 
        self.ma = 1.7866149
        self.lp = 0.12 
        self.g = 9.81
        self.ia = 0.002830449 
        self.ip = 0.000358709 

        self.kc = 0.1
        

    
    def swingup_command(self, kc, mp, lp, theta, theta_dot):
            """
            Compute the swing-up command for the pendulum.

            Parameters:
            - kc: Energy control gain
            - mp: Pendulum mass
            - lp: Pendulum length
            - theta: Current angle of the pendulum (radians, measured from downward)
            - theta_dot: Current angular velocity of the pendulum (radians/second)

            Returns:
            - u: Control input for the pendulum
            """
            # Constants
            g = 9.81  # Acceleration due to gravity (m/s^2)

            # Compute the total energy
            kinetic_energy = 0.5 * mp * (lp ** 2) * (theta_dot ** 2)
            potential_energy = -mp * g * lp * math.cos(theta)
            total_energy = kinetic_energy + potential_energy

            # Desired energy (energy at the upright position)
            desired_energy = mp * g * lp

            # Energy difference
            delta_energy = total_energy - desired_energy

            # Control input based on energy difference
            u = kc * delta_energy * theta_dot * math.cos(theta)

            return u

    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            
            pendulum_position = msg.position[pendulum_index]
            pendulum_position = ((pendulum_position + math.pi) % (2 * math.pi)) - math.pi

            pendulum_velocity = msg.velocity[pendulum_index]

            if pendulum_position >= math.pi/2 or pendulum_position <= 3*math.pi/4:
                self.effort_command = self.swingup_command(self.kc, self.mp, self.lp, pendulum_position, pendulum_velocity)
            else:
                 self.effort_command = 0

            msg_to_publish = Float64MultiArray()
            msg_to_publish.data = [self.effort_command]
            
            # Publish the message
            self.publisher_.publish(msg_to_publish)
            self.get_logger().info(f'Published value: {self.effort_command}')
            
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
