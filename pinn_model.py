"""
This module defines the Physics-Informed Neural Network (PINN) model
for solving 2-dimensional, unsteady, incompressible Navier-Stokes (NS) equations.
It includes the neural network architecture and the loss functions, which comprise
both data-driven loss and physics-based loss (from the NS equations).
"""
import os
import numpy as np
import torch
import torch.nn as nn
# It's generally good practice to import specific functions/classes if not all are needed from pyDOE
# For example: from pyDOE import lhs
# However, lhs is not directly used in this file, but in utilities.py.
# This import might be here due to an earlier version or if it was intended for direct use.
from pyDOE import lhs


# Define network structure, specified by a list of layers indicating the number of layers and neurons
# Define network structure, specified by a layer list indicating the number of layers and neurons
class PINN_Net(nn.Module):
    def __init__(self, layer_mat, mean_value, std_value, device):
        super(PINN_Net, self).__init__()
        self.X_mean = torch.from_numpy(mean_value.astype(np.float32)).to(device)
        self.X_std = torch.from_numpy(std_value.astype(np.float32)).to(device)
        self.device = device
        self.layer_num = len(layer_mat) - 1
        # Dynamically create the neural network layers
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.Initial_param()

    # 0-1 norm of input variable
    # Normalize input variables (x, y, t) using pre-calculated mean and std
    def inner_norm(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        return X_norm

    def forward_inner_norm(self, x, y, t):
        """
        Forward pass with inner normalization of inputs.
        Concatenates x, y, t, normalizes them, then passes through the network.
        """
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        X_norm = self.inner_norm(X)
        predict = self.base(X_norm)
        return predict

    def forward(self, x, y, t):
        """
        Standard forward pass without inner normalization.
        Concatenates x, y, t, then passes through the network.
        """
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # initialize
    # Initialize network parameters
    def Initial_param(self):
        """
        Initializes the weights (Xavier normal) and biases (zeros) of the linear layers.
        """
        for name, param in self.base.named_parameters():
            if name.endswith('linear.weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('linear.bias'):
                nn.init.zeros_(param)

    # derive loss for data
    # Method within the class: calculate loss for data points
    def data_mse(self, x, y, t, u, v, p):
        """
        Calculates the Mean Squared Error (MSE) loss between network predictions and true data (u, v, p).
        """
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # Method within the class: calculate loss for data points (without pressure data)
    def data_mse_without_p(self, x, y, t, u, v):
        """
        Calculates MSE loss for u and v components only (excluding pressure).
        """
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # Method within the class: predict
    def predict(self, x, y, t):
        """
        Makes predictions for u, v, and p given input coordinates x, y, t.
        Uses the standard forward pass.
        """
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation (PDE residuals)
    def equation_mse_dimensionless(self, x, y, t, Re):
        """
        Calculates the MSE loss based on the residuals of the dimensionless Navier-Stokes equations.
        This enforces the physics of the problem.
        It uses automatic differentiation to compute the necessary derivatives.
        """
        predict_out = self.forward(x, y, t)
        # Get predicted outputs u, v, p
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # Calculate partial derivatives using automatic differentiation. .sum() is used to make the output a scalar for grad.
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # Calculate the residuals of the partial differential equations
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation

    def data_mse_inner_norm(self, x, y, t, u, v, p):
        """
        Calculates data MSE loss when using inner normalization for inputs.
        """
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # Method within the class: calculate loss for data points (without pressure data), with inner normalization
    def data_mse_without_p_inner_norm(self, x, y, t, u, v):
        """
        Calculates data MSE loss for u and v (excluding p) when using inner normalization.
        """
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # Method within the class: predict, with inner normalization
    def predict_inner_norm(self, x, y, t):
        """
        Makes predictions for u, v, and p using the forward pass with inner normalization.
        """
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation (PDE residuals), with inner normalization
    def equation_mse_dimensionless_inner_norm(self, x, y, t, Re):
        """
        Calculates the MSE loss based on the residuals of the dimensionless Navier-Stokes equations,
        when the model uses inner normalization for its inputs.
        Uses automatic differentiation for derivatives.
        """
        predict_out = self.forward_inner_norm(x, y, t)
        # Get predicted outputs u, v, p
        # Note: The comment mentions u,v,w,p,k,epsilon, but the code only uses u,v,p.
        # This might be a remnant from a different model (e.g., RANS turbulence model).
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # Calculate partial derivatives using automatic differentiation. .sum() is used to make the output a scalar for grad.
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # Calculate the residuals of the partial differential equations
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation
