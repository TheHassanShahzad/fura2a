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
        self.kc = 0.1

        alpha = self.ip + self.mp * self.lp**2
        beta = self.lp * self.mp
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

        print(self.K)
        print(self.check_stable(self.Ad, self.Bd, self.K))
        
        self.X = self.X = np.zeros((4, 1), dtype=float)

        self.error_threshold = 0.5
        self.effort_command = 0

        self.u_max = 1
        self.data_array = []
        self.count = 0
        
    def discretize(self, Ac, Bc, Cc, Dc, Ts):
        # return cont2discrete((Ac, Bc, np.eye(Ac.shape[0]), 0), Ts)
        return cont2discrete((Ac, Bc, Cc, Dc), Ts)
    
    def swing_up(self, mp, lp, theta, theta_dot, g, kc):
        E = ((mp*(lp**2)*(theta_dot**2))/2) - (mp*g*lp*(cos(theta)))
        E_desired = mp*g*lp
        d_E = E-E_desired
        u = kc*d_E*theta_dot*cos(theta)
        return u
    
    def lqr(self, K, x, u_max):
        reference_state = np.array([[0], [0], [pi], [0]])
        u = -K @ (x - reference_state)
        # u = np.clip(u, -u_max, u_max)
        print(u.item())
        return u.item()
    
    def check_stable(self, Ad, Bd, K):
        A_cl = Ad - Bd @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        return eigenvalues

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
 
            # print(self.X)

            msg_to_publish = Float64MultiArray()

            # self.get_logger().info(f'Pendulum position: {pendulum_position}')

            nearest_down = round(pendulum_position/(2*pi))*2*pi
            nearest_up = (round((pendulum_position - pi) / (2 * pi)) * (2 * pi)) + pi

            d_down = abs(pendulum_position - nearest_down)
            d_up = abs(pendulum_position - nearest_up)
            
            if min(abs(pendulum_position - nearest_down), abs(pendulum_position - nearest_up)) <= self.error_threshold:

            # if (abs(pendulum_position % pi) <= self.error_threshold):


                if d_down < d_up:
                    print("down position")
                    self.effort_command = self.swing_up(self.mp, self.lp, pendulum_position, pendulum_velocity, self.g, self.kc)

                    # self.data_array.append(-1)
                else:
                    print("up position")
                    self.effort_command = self.lqr(self.K, self.X, self.u_max)


                    # self.data_array.append(1)


                # print("near equillibrium position")
 
            else:
                print("not near equillibrium points")
                # self.data_array.append(0)
                

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

            # if self.count % 10 == 0:
            #     print(self.data_array)

            # self.count += 1
            
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
