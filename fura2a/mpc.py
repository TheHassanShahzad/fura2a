#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete

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

        self.Ts = 0.02 #will set properly later
        self.N = 20

        alpha = self.ip + self.mp * self.lp**2
        beta = self.lp * self.mp * self.lp
        delta = self.ia + self.mp * self.lp**2

        # Denominator for A and B matrices
        denominator = alpha * delta - beta**2

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

        # self.Ac = np.array([
        #     [0, 1, 0, 0],
        #     [0, 0, (-self.mp * self.g * self.lp / self.ia), 0],
        #     [0, 0, 0, 1],
        #     [0, 0, ((self.ia + self.mp * (self.lp ** 2)) * self.g / (self.ia * self.ip)), 0]
        # ])

        # self.Bc = np.array([
        #     [0],
        #     [1 / self.ia],
        #     [0],
        #     [-self.lp / (self.ia * self.ip)]
        # ])


        self.Cc = np.array([[1,0,0,0], [0,0,1,0]])
        self.Dc = 0
        self.Ad, self.Bd, self.Cd, self.Dd, _ = self.discretize(self.Ac, self.Bc, self.Cc, self.Dc, self.Ts)

        self.Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*10
        self.R = np.eye(1)*10

        self.x_constraints = [
            [-np.inf, -np.inf, -np.inf, -np.inf],  # Loosen state lower bounds
            [np.inf, np.inf, np.inf, np.inf],     # Loosen state upper bounds
        ]

        self.u_constraints = [
            [-10],  # Loosen input lower bound
            [10],   # Loosen input upper bound
        ]

        self.X = np.array([[0],[0],[0],[0]])
        self.X_ref = np.array([[0],[0],[0],[0]])
        # self.F, self.G = self.build_F_G(self.N, self.Ad, self.Bd)
        self.effort_command = 0  # value to publish
        
    def discretize(self, Ac, Bc, Cc, Dc, Ts):
        return cont2discrete((Ac, Bc, np.eye(Ac.shape[0]), 0), Ts)

    # def build_F_G_U(self, N, Ad, Bd):
    #     m, n = Bd.shape[1], Ad.shape[0]
    #     F = np.zeros((n * N, n))
    #     G = np.zeros((n * N, m * N))
    #     for i in range(N):
    #         F[i*n:(i+1)*n, :] = np.linalg.matrix_power(Ad, i+1)
    #         for j in range(i+1):
    #             G[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, i-j) @ Bd
                
    #     U = cp.Variable((m * N, 1))

    #     return F, G, U
    
    def optimize(self, N, x0, xref, Ad, Bd, Q, R, x_constraints, u_constraints):
        m, n = Bd.shape[1], Ad.shape[0]  # Input and state dimensions

        # Build F and G matrices
        F = np.zeros((n * N, n))
        G = np.zeros((n * N, m * N))
        for i in range(N):
            F[i*n:(i+1)*n, :] = np.linalg.matrix_power(Ad, i+1)
            for j in range(i+1):
                G[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, i-j) @ Bd

        # Define optimization variables
        U = cp.Variable((m * N, 1))  # Control inputs

        # State trajectory
        X = F @ x0 + G @ U

        # Define cost function
        cost = 0
        for i in range(N):
            xi = X[i*n:(i+1)*n]
            ui = U[i*m:(i+1)*m]
            cost += cp.quad_form(xi - xref, Q) + cp.quad_form(ui, R)

        # Define constraints
        x_min = x_constraints[0]
        x_max = x_constraints[1]
        u_min = u_constraints[0]
        u_max = u_constraints[1]

        X_min = np.tile(x_min, N)  # Stack x_min N times for the prediction horizon
        X_max = np.tile(x_max, N)  # Stack x_max N times for the prediction horizon

        U_min = np.tile(u_min, N)  # Stack u_min N times for the control horizon
        U_max = np.tile(u_max, N)  # Stack u_max N times for the control horizon

        # CVXPY constraints
        constraints = [
            X >= X_min.reshape(-1, 1),  # State lower bounds
            X <= X_max.reshape(-1, 1),  # State upper bounds
            U >= U_min.reshape(-1, 1),  # Input lower bounds
            U <= U_max.reshape(-1, 1),  # Input upper bounds
        ]

        # Solve the optimization problem
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(verbose=False)

        # Optimal control inputs
        U_opt = U.value
        print(U_opt)
        print(" ")

        return U_opt
    
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


            # Save a value to publish (e.g., the integer part of arm_position)
            self.effort_command = float(self.optimize(self.N, self.X, self.X_ref, self.Ad, self.Bd, self.Q, self.R, self.x_constraints, self.u_constraints)[0][0])
            # self.get_logger().info(f'effort command to be published: {self.effort_command}')
            # Prepare the message to publish
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
