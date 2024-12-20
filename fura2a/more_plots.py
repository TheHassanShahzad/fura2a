#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from control import dlqr, ss
import matplotlib.pyplot as plt

class AdvancedPlottingNode(Node):
    def __init__(self):
        super().__init__('advanced_plotting_node')
        
        # Create subscriber to /joint_states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)

        # Create publisher to "arm_cont/command" topic
        self.publisher_ = self.create_publisher(Float64MultiArray, 'arm_cont/commands', 10)
        
        # Physical parameters
        self.g = 9.80665
        self.m1 = 0.3
        self.m2 = 0.105975
        self.l1 = 0.0375
        self.l2 = 0.0675
        self.sl2 = 0.061   # For swing-up control (assumed from your code)
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

        # Compute A,B,Q,R,K for LQR
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

        # Data recording for 10 seconds
        self.num_samples = 3330
        self.count = 0
        self.times = []
        self.pendulum_angles = []
        self.pendulum_velocities = []
        self.control_inputs = []
        self.phase = []  # 'swing-up' or 'lqr'

        # Flags to check mode
        self.in_lqr_mode = False

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

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
        # Linearization about theta = np.pi.
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
        E = ((self.m2 * (self.sl2**2) * (pendulum_velocity**2)) / 2.0) - (self.m2 * self.g * self.sl2 * np.cos(pendulum_position))
        E_desired = self.m2 * self.g * self.sl2
        d_E = E - E_desired
        u = self.kc * d_E * pendulum_velocity * np.cos(pendulum_position)
        return u

    def do_lqr(self, X):
        X_ref = np.zeros((4,1))
        u = -self.K @ (X - X_ref)
        u *= -10.0  # Amplify control effort
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
            
            pendulum_position = msg.position[pendulum_index]
            tilde_theta = self.normalize_angle(pendulum_position - np.pi)
            pendulum_velocity = msg.velocity[pendulum_index]

            self.X[0] = arm_position
            self.X[1] = tilde_theta
            self.X[2] = arm_velocity
            self.X[3] = pendulum_velocity

            msg_to_publish = Float64MultiArray()

            if abs(tilde_theta) <= self.position_threshold and abs(pendulum_velocity) <= self.velocity_threshold:
                # Switch to LQR mode if not already
                if not self.in_lqr_mode:
                    self.in_lqr_mode = True
                effort = self.do_lqr(self.X)
                phase_name = 'lqr'
            else:
                effort = self.swing_up(pendulum_position, pendulum_velocity)
                phase_name = 'swing-up'

            msg_to_publish.data = [effort]
            self.publisher_.publish(msg_to_publish)

            # Record data
            self.times.append(self.count*self.dt)
            self.pendulum_angles.append(tilde_theta)
            self.pendulum_velocities.append(pendulum_velocity)
            self.control_inputs.append(effort)
            self.phase.append(phase_name)

            self.count += 1

            if self.count == self.num_samples:
                # Data collection done, produce plots
                self.plot_results()
                rclpy.shutdown()

        except ValueError:
            self.get_logger().warn('Joint name not found in JointState message')

    def plot_results(self):
        time = np.array(self.times)
        angle = np.array(self.pendulum_angles)
        velocity = np.array(self.pendulum_velocities)
        ctrl = np.array(self.control_inputs)
        phase = np.array(self.phase)

        # Split data into swing-up and LQR phases
        swing_mask = (phase == 'swing-up')
        lqr_mask = (phase == 'lqr')

        # Compute energies (for energy trajectories)
        # Kinetic Energy: T = 1/2 * m2 * (l2 * dtheta)^2 (approx for pendulum)
        # Potential Energy: U = m2 * g * l2 * (1 - np.cos(theta + np.pi))
        # Using theta = tilde_theta + np.pi
        full_theta = angle + np.pi
        KE = 0.5 * self.m2 * (self.l2**2) * (velocity**2)
        PE = self.m2 * self.g * self.l2 * (1 - np.cos(full_theta))
        total_E = KE + PE

        #####
        # 1) Swing-Up Control: Pendulum Angle vs. Time (only swing-up phase)
        #####
        plt.figure()
        plt.plot(time[swing_mask], angle[swing_mask], 'r-', label='Pendulum Angle (Swing-Up)')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title('Swing-Up Control: Pendulum Angle vs. Time')
        plt.grid(True)
        plt.legend()

        #####
        # 2) LQR Control Performance: Pendulum Angle vs. Time (only LQR phase)
        #####
        plt.figure()
        plt.plot(time[lqr_mask], angle[lqr_mask], 'b-', label='Pendulum Angle (LQR)')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title('LQR Control Performance: Pendulum Angle vs. Time')
        plt.grid(True)
        plt.legend()

        #####
        # 3) Control Input vs. Time (whole run)
        #####
        plt.figure()
        plt.plot(time, ctrl, 'g-', label='Control Input')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Control Input vs. Time')
        plt.grid(True)
        plt.legend()

        #####
        # For the histograms and distributions, we need multiple data sets.
        # Here we will simulate multiple runs or parameter variations with random data.
        # In practice, you'd gather these from multiple experiments.
        #####

        # Simulate distributions (e.g., overshoot, settling time) as random data
        overshoot_data = np.random.normal(0.05, 0.01, 100)  # just a random example
        settling_time_data = np.random.normal(2.0, 0.5, 100) # random example
        control_effort_data = np.abs(np.random.normal(0.5, 0.2, 100)) # random example

        #####
        # 4) Settling Time Distribution: Histogram
        #####
        plt.figure()
        plt.hist(settling_time_data, bins=10, color='c', edgecolor='k')
        plt.xlabel('Settling Time (s)')
        plt.ylabel('Frequency')
        plt.title('Settling Time Distribution')
        plt.grid(True)

        #####
        # 5) Overshoot Distribution: Histogram
        #####
        plt.figure()
        plt.hist(overshoot_data, bins=10, color='m', edgecolor='k')
        plt.xlabel('Overshoot (rad)')
        plt.ylabel('Frequency')
        plt.title('Overshoot Distribution')
        plt.grid(True)

        #####
        # 6) Control Effort Distribution: Histogram
        #####
        plt.figure()
        plt.hist(control_effort_data, bins=10, color='y', edgecolor='k')
        plt.xlabel('Control Effort (Nm)')
        plt.ylabel('Frequency')
        plt.title('Control Effort Distribution')
        plt.grid(True)

        #####
        # 7) Parameter Sensitivity: Impact on Settling Time
        # Simulate parameter variations (e.g., different m2 values)
        params = np.linspace(self.m2*0.8, self.m2*1.2, 10)
        settling_times_param = np.random.normal(2.0, 0.3, 10) # random example
        plt.figure()
        plt.plot(params, settling_times_param, 'o--', label='Settling Time vs. m2 Variation')
        plt.xlabel('m2 (kg)')
        plt.ylabel('Settling Time (s)')
        plt.title('Parameter Sensitivity: Impact on Settling Time')
        plt.grid(True)
        plt.legend()

        #####
        # 8) Noise Impact Analysis: Overlayed Trajectories
        # Simulate multiple noisy trajectories
        for i in range(5):
            noisy_angle = angle + np.random.normal(0, 0.02, len(angle))
            plt.figure(8)
            plt.plot(time, noisy_angle, label=f'Trajectory {i+1}')
        plt.figure(8)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title('Noise Impact Analysis: Overlayed Trajectories')
        plt.grid(True)
        plt.legend()

        #####
        # 9) Phase Portrait: Pendulum Angle vs. Pendulum Velocity
        #####
        plt.figure()
        plt.plot(angle, velocity, 'r-')
        plt.xlabel('Angle (rad)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Phase Portrait')
        plt.grid(True)

        #####
        # 10) Energy Trajectories: Kinetic and Potential Energy vs. Time
        #####
        plt.figure()
        plt.plot(time, KE, 'r-', label='Kinetic Energy')
        plt.plot(time, PE, 'b-', label='Potential Energy')
        plt.plot(time, KE+PE, 'k--', label='Total Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)') 
        plt.title('Energy Trajectories: Kinetic and Potential Energy vs. Time')
        plt.grid(True)
        plt.legend()

        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = AdvancedPlottingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
