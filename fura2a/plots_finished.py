#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from math import pi, cos
import numpy as np
from control import dlqr, ss
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

        # Create publisher to "arm_cont/command" topic
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            'arm_cont/commands',
            10)
        
        # Physical parameters (example values from your URDF and code)
        self.g = 9.80665
        self.m1 = 0.3
        self.m2 = 0.105975
        self.l1 = 0.0375
        self.l2 = 0.0675
        self.sl2 = 0.061   # For swing-up control
        self.L1 = 0.08
        self.L2 = 0.135
        self.b1 = 0.0001
        self.b2 = 0.0003

        # Moments of inertia
        self.J1 = 1.42675159e-04
        self.J2 = 0.000162
        
        # From simplifications
        self.J2_hat = self.J2 + self.m2 * (self.l2**2)
        self.J0_hat = self.J1 + self.m1 * (self.l1**2) + self.m2 * (self.L1**2)

        self.dt = 1/333.0
        
        # Weights for LQR
        self.theta1_weight = 0.0
        self.theta2_weight = 100.0
        self.dtheta1_weight = 10.0
        self.dtheta2_weight = 0.0
        self.u_weight = 0.01

        # Thresholds for switching to LQR
        self.position_threshold = 0.35
        self.velocity_threshold = 100000

        # Get linearized A and B for the inverted equilibrium (theta = pi)
        self.A, self.B, self.Q, self.R = self.get_ABQR_inverted()
        self.is_controllable()
        self.K, self.S, self.E = self.get_KSE_d()

        self.X = np.zeros((4, 1), dtype=float)
        self.kc = 0.075  # Swing-up gain

        print("J2_hat is", self.J2_hat)
        print("J0_hat is", self.J0_hat) 
        print("A is", self.A)
        print("B is", self.B)
        print("Q is", self.Q)
        print("R is", self.R)
        print("K is", self.K)
        print("S is", self.S)
        print("E is", self.E)

        # Data recording
        self.num_samples = 3330  # 10s * 333Hz
        self.count = 0
        self.pendulum_positions = []
        self.pendulum_velocities = []
        self.control_inputs = []
        self.times = []

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi].
        """
        return (angle + pi) % (2 * pi) - pi

    def is_controllable(self):
        n = self.A.shape[0]
        C = self.B
        for i in range(1, n):
            C = np.hstack((C, np.linalg.matrix_power(self.A, i).dot(self.B)))
        rank = np.linalg.matrix_rank(C)
        if rank == n:
            print("The system is controllable.")
        else:
            print("The system is NOT controllable.")

    def get_ABQR_inverted(self):
        # Linearization about theta = pi.
        # At the top, cos(pi) = -1.
        denominator = self.J0_hat * self.J2_hat - (self.m2**2) * (self.L1**2) * (self.l2**2)
        
        A32 = -(self.g * (self.m2**2) * (self.l2**2) * self.L1) / denominator
        A42 = -(self.g * self.m2 * self.l2 * self.J0_hat) / denominator

        A33 = (-self.b1 * self.J2_hat) / denominator
        A34 = ( self.b2 * self.m2 * self.l2 * self.L1) / denominator
        A43 = ( self.b1 * self.m2 * self.l2 * self.L1) / denominator
        A44 = (-self.b2 * self.J0_hat) / denominator

        B31 = ( self.J2_hat ) / denominator
        B41 = ( self.m2 * self.L1 * self.l2 ) / denominator

        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, A32, A33, A34],
            [0.0, A42, A43, A44]
        ])

        B = np.array([
            [0.0],
            [0.0],
            [B31],
            [B41]
        ])

        Q = np.diag([self.theta1_weight, self.theta2_weight, self.dtheta1_weight, self.dtheta2_weight])
        R = np.array([[self.u_weight]])

        return A, B, Q, R

    def get_KSE_d(self):
        C = np.identity(4)
        D = np.zeros((4,1))
        dsys = ss(self.A, self.B, C, D, self.dt)
        K, S, E = dlqr(dsys, self.Q, self.R)
        return K, S, E

    def swing_up(self, pendulum_position, pendulum_velocity):
        E = ((self.m2 * (self.sl2**2) * (pendulum_velocity**2)) / 2.0) - (self.m2 * self.g * self.sl2 * cos(pendulum_position))
        E_desired = self.m2 * self.g * self.sl2
        d_E = E - E_desired
        u = self.kc * d_E * pendulum_velocity * cos(pendulum_position)
        return u

    def do_lqr(self, X):
        X_ref = np.zeros((4,1))
        u = -self.K @ (X - X_ref) 
        u *= -10.0  # Amplify control effort
        # print("LQR control effort is", u.item())
        return u.item()

    def listener_callback(self, msg):
        if self.count >= self.num_samples:
            # Already captured enough data, no more processing
            return
        
        try:
            arm_index = msg.name.index('arm_joint')
            pendulum_index = msg.name.index('pendulum_joint')
            
            arm_position = msg.position[arm_index]
            arm_velocity = msg.velocity[arm_index]
            
            # tilde_theta = theta - pi
            pendulum_position = msg.position[pendulum_index]
            tilde_theta = self.normalize_angle(pendulum_position - pi)
            pendulum_velocity = msg.velocity[pendulum_index]

            # State vector: [phi, tilde_theta, dphi, dtilde_theta]
            self.X[0] = arm_position
            self.X[1] = tilde_theta
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            msg_to_publish = Float64MultiArray()

            if abs(tilde_theta) <= self.position_threshold and abs(pendulum_velocity) <= self.velocity_threshold:
                self.effort_command = self.do_lqr(self.X)
            else:
                self.effort_command = self.swing_up(pendulum_position, pendulum_velocity)
            
            msg_to_publish.data = [self.effort_command]
            self.publisher_.publish(msg_to_publish)

            # Record data
            self.pendulum_positions.append(tilde_theta)
            self.pendulum_velocities.append(pendulum_velocity)
            self.control_inputs.append(self.effort_command)
            self.times.append(self.count * self.dt)

            self.count += 1

            # Once we have collected all data, plot and shutdown
            if self.count == self.num_samples:
                self.plot_results()
                # Stop the node after plotting
                rclpy.shutdown()

        except ValueError:
            self.get_logger().warn('Joint name not found in JointState message')
    
    def plot_results(self):
        time = np.array(self.times)
        pos = np.array(self.pendulum_positions)
        vel = np.array(self.pendulum_velocities)
        ctrl = np.array(self.control_inputs)

        # Plot pendulum position
        plt.figure()
        plt.plot(time, pos, 'r-', label='Pendulum Position (rad)')
        plt.xlabel('Time (s)')
        plt.ylabel('Pendulum Position (rad)')
        plt.title('Pendulum Position vs Time')
        plt.grid(True)
        plt.legend()

        # Plot pendulum velocity
        plt.figure()
        plt.plot(time, vel, 'b-', label='Pendulum Velocity (rad/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Pendulum Velocity (rad/s)')
        plt.title('Pendulum Velocity vs Time')
        plt.grid(True)
        plt.legend()

        # Plot control input
        plt.figure()
        plt.plot(time, ctrl, 'g-', label='Control Input (Nm)')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input (Nm)')
        plt.title('Control Input vs Time')
        plt.grid(True)
        plt.legend()

        # Plot all together
        plt.figure()
        plt.plot(time, pos, 'r-', label='Pos (rad)')
        plt.plot(time, vel, 'b-', label='Vel (rad/s)')
        plt.plot(time, ctrl, 'g-', label='Ctrl (Nm)')
        plt.xlabel('Time (s)')
        plt.title('Pendulum Position, Velocity, and Control Input')
        plt.grid(True)
        plt.legend()

        plt.show()


def main(args=None):
    rclpy.init(args=args)
    joint_state_processor = JointStateProcessor()
    rclpy.spin(joint_state_processor)
    # Cleanup
    joint_state_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
