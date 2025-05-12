"""
Main training script for the Physics-Informed Neural Network (PINN)
for solving 2D unsteady Navier-Stokes equations.

This script handles:
- Configuration of training parameters.
- Data loading and preprocessing.
- PINN model initialization.
- Optimizer and learning rate scheduler setup.
- The main training loop, including loss calculation, backpropagation, and model saving.
- Validation during training.
"""
from utilities import *
import time
from argparse import Namespace
from validation_func import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_config = Namespace(
    write_path='./write',
    loss_path='./write/loss.csv',
    evaluate_path='./write/RL2.csv',
    data_path='./cylinder_Re3900_36points_100snaps.mat',
    real_path='./cylinder_Re3900_ke_all_100snaps.mat',
    Re=3900,
    dimension=2 + 1,  # Spatial dimensions (x, y) + time (t)
    outer_epoch=5000, # Total number of training epochs
    inner_epochs=1,   # Number of inner loops for training within each epoch (usually 1 for this setup)
    save_interval=100,# Interval for saving the model and performing validation
    number_eqa=1000000, # Number of collocation points for PDE residual calculation
    inner_Norm="with_norm", # "with_norm" or "no_norm", specifies if inner normalization is used
    optimizer='adam', # Optimizer type: 'adam' or 'sgd'
    scheduler='exp',
    batch_size=10000,
    learning_rate=1e-3,
    hidden_layers=10,
    layer_neurons=32,
    weight_of_data=1,
    weight_of_eqa=1,
    debug_key=1,
)


def build_optimizer(network, optimizer_name, scheduler_name, learning_rate):
    """
    Builds and returns an optimizer and a learning rate scheduler.
    Args:
        network (torch.nn.Module): The neural network model.
        optimizer_name (str): Name of the optimizer ('adam', 'sgd').
        scheduler_name (str): Name of the scheduler ('exp', 'fix').
        learning_rate (float): Initial learning rate.
    """
    # Default optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # Optimizer based on hyperparameter search
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    if scheduler_name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif scheduler_name == "fix":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return optimizer, scheduler


def validation(pinn_net, real_path, inner_Norm, filename_RL2):
    """
    Performs validation by computing the relative L2 norm of the predictions.
    Args:
        pinn_net (torch.nn.Module): The PINN model to validate.
        real_path (str): Path to the file containing the true data for comparison.
        inner_Norm (str): Specifies if the model uses inner normalization.
        filename_RL2 (str): Path to the CSV file to save the L2 norm results.
    """
    RL2_u, RL2_v, RL2_p = compute_L2_norm(real_path, pinn_net, device, norm_status=inner_Norm)
    RL2_u_value = RL2_u.reshape(1, 1)
    RL2_v_value = RL2_v.reshape(1, 1)
    RL2_p_value = RL2_p.reshape(1, 1)
    RL2_set = np.concatenate((RL2_u_value, RL2_v_value, RL2_p_value), 1).reshape(1, -1)
    ReL2_save = pd.DataFrame(RL2_set)
    ReL2_save.to_csv(filename_RL2, index=False, header=False, mode='a')
    return RL2_u_value, RL2_v_value, RL2_p_value


def train():
    """
    Main training function.
    Sets up the experiment, loads data, initializes the model,
    and runs the training loop.
    """
    if device.type == 'cpu':
        print("Warning: Training on CPU. CUDA is recommended for faster training.")

    # important parameters
    # Important parameters from the configuration
    data_path = train_config.data_path  # Training data file
    Re = train_config.Re  # Reynolds number
    dimension = train_config.dimension
    real_path = train_config.real_path # File with full true data for validation
    write_path = train_config.write_path # Directory to save model checkpoints and outputs
    evaluate_path = train_config.evaluate_path # CSV file to save L2 evaluation results
    loss_path = train_config.loss_path # CSV file to save loss history
    inner_epochs = train_config.inner_epochs
    save_interval = train_config.save_interval  # Model saving period
    outer_epochs = train_config.outer_epoch  # Number of training epochs
    number_eqa = train_config.number_eqa # Number of collocation points
    debug_key = train_config.debug_key # Flag to enable/disable debug print statements
    inner_Norm = train_config.inner_Norm # Normalization strategy

    # Hyperparameters for network and training
    layer_mat = [dimension] + train_config.hidden_layers * [train_config.layer_neurons] + [3]
    learning_rate = train_config.learning_rate  # Learning rate
    batch_size = train_config.batch_size
    weight_of_data = train_config.weight_of_data # Weight for the data loss component
    weight_of_eqa = train_config.weight_of_eqa   # Weight for the PDE/equation loss component
    optimizer_name = train_config.optimizer      # Optimizer choice
    scheduler_name = train_config.scheduler      # Learning rate scheduler choice

    # pretraining-loading data
    # loading data points
    # loading collocation points
    # only load once
    # Pre-training loading (only once)
    # Load data points and collocation points
    True_dataset_batches, Eqa_points_batches, iter_num, low_bound, up_bound = pre_train_loading(
        data_path,
        dimension,
        number_eqa,
        batch_size, )
    data_mean, data_std = load_data_feature(data_path)
    # Initialize the PINN model
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)

    # 优化器和学习率衰减设置- optimizer and learning rate schedule
    optimizer_all, scheduler_all = build_optimizer(pinn_net, optimizer_name, scheduler_name, learning_rate)

    start = time.time()
    if not os.path.exists(write_path):
        # Create directory for saving outputs if it doesn't exist
        os.mkdir(write_path)
    # Main training loop
    for EPOCH in range(outer_epochs):

        # Train simultaneously on data points and collocation points
        # Each iteration trains on all data within the batch
        # No inner normalization layer
        if inner_Norm == "no_norm":
            loss_sum, loss_data, loss_eqa = train_data_whole(inner_epochs, pinn_net, optimizer_all, scheduler_all,
                                                             iter_num, True_dataset_batches, Eqa_points_batches,
                                                             Re, weight_of_data, weight_of_eqa, EPOCH, debug_key,
                                                             device)
        # With inner normalization layer
        else:
            loss_sum, loss_data, loss_eqa = train_data_whole_inner_norm(inner_epochs, pinn_net, optimizer_all,
                                                                        scheduler_all,
                                                                        iter_num, True_dataset_batches,
                                                                        Eqa_points_batches, Re, weight_of_data,
                                                                        weight_of_eqa, EPOCH,
                                                                        debug_key, device)

        # Record loss
        loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_path)  # Record sub-network loss

        # Save model at every save_interval epoch
        # Evaluate model at every save_interval epoch
        if not os.path.exists(write_path): # This check is redundant if done before the loop
            os.makedirs(write_path)
        if ((EPOCH + 1) % save_interval == 0) | (EPOCH == 0):
            dir_name = write_path + '/step' + str(EPOCH + 1)
            os.makedirs(dir_name, exist_ok=True)
            torch.save(pinn_net.state_dict(), dir_name + '/NS_model_train.pt')
            print(f'Model saved at step {EPOCH + 1}.')
            valid_u, valid_v, valid_p = validation(pinn_net, real_path, inner_Norm, evaluate_path)
            print(f'Model evaluated at step {EPOCH + 1}.')

    end = time.time()
    print("Time used: ", end - start)
    return

# Entry point of the script
if __name__ == '__main__':
    train()
