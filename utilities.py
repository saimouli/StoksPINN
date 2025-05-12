"""
This module provides core utility functions for the PINN model.
It includes functions for data loading, preprocessing, batching,
and the main training loops for the neural network.
"""
import numpy as np
import scipy
import pandas as pd
from pinn_model import *
# It's generally good practice to import specific functions/classes if not all are needed from pyDOE
# For example: from pyDOE import lhs
from pyDOE import lhs # Assuming lhs is used for Latin Hypercube Sampling

# load data feature for inner normalization method
# Load mean and standard deviation of data, suitable for inner normalization method
def load_data_feature(filename):
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    temp = np.concatenate((x, y, t), 1)
    data_mean = np.mean(temp, axis=0).reshape(1, -1)
    data_std = np.std(temp, axis=0).reshape(1, -1)
    return data_mean, data_std


# load data points
# Load data points for the network
def load_data_points(filename):
    """
    Loads raw data points (x, y, t, u, v, p) from a .mat file.
    Calculates lower and upper bounds for x, y, t.
    Converts numpy arrays to PyTorch tensors.
    """
    # Read raw data
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    low_bound = np.array([np.min(x), np.min(y), np.min(t)]).reshape(1, -1)
    up_bound = np.array([np.max(x), np.max(y), np.max(t)]).reshape(1, -1)
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.min(temp, 0)
    feature_mat[1, :] = np.max(temp, 0)
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


# load collocation points
# Load collocation points using Latin Hypercube Sampling (LHS)
def load_equation_points_lhs(low_bound, up_bound, dimension, points):
    """
    Generates collocation points for enforcing the PDE using Latin Hypercube Sampling.
    These points are sampled within the specified lower and upper bounds.
    Args:
        dimension (int): Number of dimensions for LHS (e.g., 3 for x, y, t).
    """
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    Eqa_points = torch.from_numpy(eqa_xyzt).float()
    Eqa_points = Eqa_points[torch.randperm(Eqa_points.size(0))]
    return Eqa_points


# split batch and automatically fill batch size
# Batch splitting and automatic padding
def batch_split(Set, iter_num, dim=0):
    """
    Splits a dataset into a specified number of batches.
    If the number of generated batches is less than iter_num, it duplicates existing batches to match iter_num.
    """
    batches = torch.chunk(Set, iter_num, dim=dim)
    # Automatic padding
    num_of_batches = len(batches)
    if num_of_batches == 1:
        batches = batches * iter_num
        return batches
    if num_of_batches < iter_num:
        for i in range(iter_num - num_of_batches):
            index = i % num_of_batches
            add_tuple = batches[-(index + 2):-(index + 1)]
            batches = batches + add_tuple
        return batches
    else:
        return batches


# preprocessing before training
# Preprocessing before training: data splitting into batches
def pre_train_loading(filename_data, dimension, N_eqa, batch_size):
    """
    Performs pre-training loading steps:
    1. Loads true data points.
    2. Shuffles the true dataset.
    3. Loads collocation points (for PDE residuals).
    4. Splits both true data and collocation points into batches.
    """
    # load data points(only once)
    # Load true data points (only once)
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, low_bound, up_bound = load_data_points(filename_data)
    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # Shuffle
    else:
        true_dataset = None
    # load collocation points(only once)
    # Load collocation points (only once)
    eqa_points = load_equation_points_lhs(low_bound, up_bound, dimension, N_eqa)
    eqa_points_batches = torch.split(eqa_points, batch_size, dim=0)
    iter_num = len(eqa_points_batches)
    if true_dataset is not None:
        true_dataset_batches = batch_split(true_dataset, iter_num)
    else:
        # Handle the case where true_dataset is None, perhaps by creating empty batches or raising an error
        true_dataset_batches = [None] * iter_num # Or some other appropriate handling
    return true_dataset, eqa_points_batches, iter_num, low_bound, up_bound

# train data points, collocation points--with batch training
# Simultaneously train data points and collocation points --- with batching
def train_data_whole(inner_epochs, pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                     Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for epoch in range(inner_epochs):
        for batch_iter in range(iter_num):
            optimizer_all.zero_grad()
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
            mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re)
            # calculate loss
            # Calculate loss function
            loss = weight_data * mse_data + weight_eqa * mse_equation
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_sum = loss.cpu().data.numpy()
                loss_data = mse_data.cpu().data.numpy()
                loss_eqa = mse_equation.cpu().data.numpy()
                # print status
                # Output status
                if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
        scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# train with inner normalization
# Training with network-embedded normalization layer
def train_data_whole_inner_norm(inner_epochs, pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                                Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for epoch in range(inner_epochs):
        for batch_iter in range(iter_num):
            optimizer_all.zero_grad()
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            mse_data = pinn_example.data_mse_inner_norm(x_train, y_train, t_train, u_train, v_train, p_train)
            mse_equation = pinn_example.equation_mse_dimensionless_inner_norm(x_eqa, y_eqa, t_eqa, Re)
            # calculate loss
            # Calculate loss function
            loss = weight_data * mse_data + weight_eqa * mse_equation
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_sum = loss.cpu().data.numpy()
                loss_data = mse_data.cpu().data.numpy()
                loss_eqa = mse_equation.cpu().data.numpy()
                # print status
                # Output status
                if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
        scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# record loss
# Record loss values
def record_loss_local(loss_sum, loss_data, loss_eqa, filename_loss):
    """
    Records the total loss, data loss, and equation loss to a CSV file.
    """
    loss_sum_value = loss_sum.reshape(1, 1)
    loss_data_value = loss_data.reshape(1, 1)
    loss_eqa_value = loss_eqa.reshape(1, 1)
    loss_set = np.concatenate((loss_sum_value, loss_data_value, loss_eqa_value), 1).reshape(1, -1)
    loss_save = pd.DataFrame(loss_set)
    loss_save.to_csv(filename_loss, index=False, header=False, mode='a')
    return loss_set
