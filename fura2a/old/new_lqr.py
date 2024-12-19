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
        
        self.m1 = 1.7866149
        self.m2 = 0.0893979 
        self.l2 = 0.06
        self.L1 = 0.075
        self.g = 9.81
        self.b1 = 0
        self.b2 = 0
        # self.J0 = 0.00283041
        # self.J2 = 0.000322832
        self.J0 = 0.012765
        self.J2 = 0.000479
        self.kc = 0.075
        self.Ts = 0.00303

        self.A31, self.A32, self.A33, self.A34 = 0, 0, 0, 0
        self.A41, self.A42, self.A43, self.A44 = 0, 0, 0, 0
        self.B31, self.B32, self.B41, self.B42 = 0, 0, 0, 0

        self.compute_linearized_matrices()

        self.Ac = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [self.A31, self.A32, self.A33, self.A34],
            [self.A41, self.A42, self.A43, self.A44]
        ])
        self.Bc = np.array([
            [0, 0],
            [0, 0],
            [self.B31, self.B32],
            [self.B41, self.B42]
        ])

        self.Cc = np.array([[1,0,0,0], [0,0,1,0]])
        self.Dc = 0

        self.Ad, self.Bd, self.Cd, self.Dd, _ = self.discretize(self.Ac, self.Bc, self.Ts)

        self.Q = np.diag([1, 10, 1, 10])
        self.R = np.diag([5,5])

        self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ (self.Bd.T @ self.P @ self.Ad)

        print(self.K)
        print(list(self.check_stable(self.Ad, self.Bd, self.K)))
        
        self.X = self.X = np.zeros((4, 1), dtype=float)

        self.error_threshold = 0.2
        self.effort_command = 0

        self.u_max = 2
        self.data_array = []
        self.count = 0
        
    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_linearized_matrices(self):
        denominator = (self.J0 * self.J2 - self.m2**2 * self.L1**2 * self.l2**2)
        if denominator == 0:
            raise ValueError("Denominator cannot be zero; check input parameters.")

        # A Matrix Elements
        self.A31 = 0
        self.A32 = (self.g * self.m2**2 * self.l2**2 * self.L1) / denominator
        self.A33 = -self.b1 * self.J2 / denominator  # Will be zero since b1 = 0
        self.A34 = -self.b2 * self.m2 * self.l2 * self.L1 / denominator  # Zero since b2 = 0

        self.A41 = 0
        self.A42 = (self.g * self.m2 * self.l2 * self.J0) / denominator
        self.A43 = -self.b1 * self.m2 * self.l2 * self.L1 / denominator  # Zero since b1 = 0
        self.A44 = -self.b2 * self.J0 / denominator  # Zero since b2 = 0

        # B Matrix Elements
        self.B31 = self.J2 / denominator
        self.B32 = self.m2 * self.L1 * self.l2 / denominator
        self.B41 = self.m2 * self.L1 * self.l2 / denominator
        self.B42 = self.J0 / denominator


    def discretize(self, Ac, Bc, Ts):
        return cont2discrete((Ac, Bc, np.eye(Ac.shape[0]), 0), Ts)
    
    def swing_up(self, mp, lp, theta, theta_dot, g, kc):
        E = ((mp * (lp**2) * (theta_dot**2)) / 2) - (mp * g * lp * (cos(theta)))
        E_desired = mp * g * lp
        d_E = E - E_desired
        u = kc * d_E * theta_dot * cos(theta)
        return u


    # def lqr(self, K, x, u_max=np.Inf):
    #     reference_state = np.array([[0], [pi], [0], [0]])
    #     # u = -K @ (x - reference_state)
    #     # u = np.clip(u[0], -u_max, u_max)
    #     # print(u[0].item())
    #     # return -u[0].item()

    #     error_state = self.X.copy()
    #     error_state[1] = self.normalize_angle(self.X[1] - np.pi)  # Normalize pendulum angle error
    #     u = -self.K @ (error_state - reference_state)
    #     return u

    def lqr(self, K, x, u_max=np.Inf):
        """
        Computes the LQR control effort for the Furuta pendulum.

        Args:
            K (np.array): LQR gain matrix (2x4).
            x (np.array): Current state vector (4x1).
            u_max (float): Maximum allowable control effort.

        Returns:
            float: Control effort for the actuator (scalar).
        """
        reference_state = np.array([[0], [np.pi], [0], [0]])  # Upright reference state

        # Normalize the pendulum angle error (second state)
        error_state = x.copy()
        error_state[1] = self.normalize_angle(x[1] - np.pi)

        # Compute control input: u = -K @ (x - reference_state)
        u_vector = -K @ (error_state - reference_state)  # Result is 2x1

        # Extract the first control effort (assumed for the arm motor)
        u_scalar = u_vector[0, 0]

        # Clip the control effort to avoid excessive commands
        u_scalar = np.clip(u_scalar, -u_max, u_max)

        # Optionally, print for debugging
        print(f"LQR control effort: {u_scalar}")

        return u_scalar


    def check_stable(self, Ad, Bd, K):
        A_cl = Ad - Bd @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        yield eigenvalues

        if max(abs(eigenvalues)) < 1:
            yield True
        else:
            yield False
    
    def blend_control(self, swing_up_effort, lqr_effort, alpha):
        """
        Blend swing-up and LQR control efforts smoothly.
        Args:
            swing_up_effort: Control effort from the swing-up controller.
            lqr_effort: Control effort from the LQR controller.
            alpha: Blending factor (0 to 1). Closer to 1 means favor LQR.
        Returns:
            Blended control effort.
        """
        return alpha * lqr_effort + (1 - alpha) * swing_up_effort


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

            # if abs(pi - abs(pendulum_position)) <= self.error_threshold and abs(pendulum_velocity) < 0.5:
            #     print("start lqr")
            #     self.effort_command = self.lqr(self.K, self.X, self.u_max)
            # else:
            #     print("swing-up active")
            #     self.effort_command = self.swing_up(self.m2, self.l2, pendulum_position, pendulum_velocity, self.g, self.kc)


            # if abs(pi - abs(pendulum_position)) <= self.error_threshold and abs(pendulum_velocity) < 0.5:
            #     print("LQR Active")
            #     lqr_effort = self.lqr(self.K, self.X, self.u_max)
            # else:
            #     print("Swing-Up Active")
            #     lqr_effort = 0  # Default to zero effort when LQR not active
                
            # swing_up_effort = self.swing_up(self.m2, self.l2, pendulum_position, pendulum_velocity, self.g, self.kc)

            # # Smoothly blend control efforts
            # alpha = min(1.0, max(0.0, 1 - abs(pi - abs(pendulum_position)) / self.error_threshold))
            # self.effort_command = self.blend_control(swing_up_effort, lqr_effort, alpha)

            # if abs(pi - abs(pendulum_position)) <= self.error_threshold and abs(pendulum_velocity) < 0.1:
            #     print("start lqr")
            #     self.effort_command = self.lqr(self.K, self.X, self.u_max)
            # else:
            #     print("swing-up active")
            #     self.effort_command = self.swing_up(self.m2, self.l2, pendulum_position, pendulum_velocity, self.g, self.kc)

            if (pi - abs(pendulum_position)) <= self.error_threshold:
                print("start lqr")
                self.effort_command = self.lqr(self.K, self.X, self.u_max)
            elif (pi - abs(pendulum_position)) >= pi/2:
                print("swing up")
                self.effort_command = self.swing_up(self.m2, self.l2, pendulum_position, pendulum_velocity, self.g, self.kc)
            else:
                print("do nothing")
                self.effort_command = 0.0 
            # self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
            msg_to_publish.data = [self.effort_command]
            
            # Publish the message
            self.publisher_.publish(msg_to_publish)

            # if self.count % 10 == 0:
            #     rounded_position = round(pendulum_position, 3)
            #     self.data_array.append(rounded_position)
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
