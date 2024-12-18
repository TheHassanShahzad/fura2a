#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
import math

import matplotlib.pyplot as plt



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

        self.X = self.X = np.zeros((4, 1), dtype=float)

        self.ks = 0.05
        self.u_max = 1.5

        self.count = 0
        self.max_count = 330*13
        self.data = [[],[]]

    def limit_minus_pi_pi(self, angle):
        angle = math.fmod(angle, 2 * math.pi)  # Wrap angle within [-2*pi, 2*pi]
        if angle > math.pi:
            angle -= 2 * math.pi  # Adjust if angle > pi
        elif angle < -math.pi:
            angle += 2 * math.pi  # Adjust if angle < -pi
        return angle
    

    def swingup(self, position, velocity):
        theta = self.limit_minus_pi_pi(position - math.pi)
        E0 = self.m2 * self.g * self.l2
        KE = 0.5 * self.m2 * (self.l2**2) * (velocity**2) 
        PE = self.m2 * self.g * self.l2 * math.cos(theta)
        E = KE + PE
        d_E = E0 - E

        # Scale the torque smoothly based on energy difference
        torque = self.ks * d_E * velocity * math.cos(theta)

        return torque
        # return np.clip(torque, -self.u_max, self.u_max)

    def listener_callback(self, msg):
        # Extract positions and velocities
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            pendulum_position = msg.position[pendulum_index]
            normalised_pendulum_position = self.limit_minus_pi_pi(pendulum_position)
            pendulum_velocity = msg.velocity[pendulum_index]

            self.X[0] = arm_position
            self.X[1] = normalised_pendulum_position
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            self.data[0].append(pendulum_position)
            self.data[1].append(normalised_pendulum_position)

            self.effort_command = self.swingup(pendulum_position, pendulum_velocity)

            msg_to_publish = Float64MultiArray()
            msg_to_publish.data = [self.effort_command]
            self.publisher_.publish(msg_to_publish)

            if self.count == self.max_count:
                pos = self.data[0]
                norm_pos = self.data[1]

                plt.figure(figsize=(8, 5))
                plt.plot(pos, label="not normalised")
                plt.plot(norm_pos, label="normalised")

                # Adding labels and legend
                plt.xlabel("Time Steps")
                plt.ylabel("Radians")
                plt.title("normalising angles vs without")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()

            print(self.count)
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
