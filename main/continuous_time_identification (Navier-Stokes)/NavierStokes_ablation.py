import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import json
from datetime import datetime
import argparse

np.random.seed(1234)
torch.manual_seed(1234)

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='tanh'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.linear(x)
        if self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'elu':
            x = torch.nn.functional.elu(x)
        elif self.activation == 'leaky_relu':
            x = torch.nn.functional.leaky_relu(x, 0.01)
        # else: no activation (linear)
        return x

class NeuralNet(nn.Module):
    def __init__(self, layers, lb, ub, device, activation='tanh'):
        super(NeuralNet, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        self.activation = activation
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            if i < len(layers) - 2:
                self.layers.append(DenseLayer(layers[i], layers[i+1], activation=activation))
            else:
                self.layers.append(DenseLayer(layers[i], layers[i+1], activation=None))
    
    def forward(self, x):
        # Normalize inputs
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        for layer in self.layers:
            H = layer(H)
        
        return H

class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers, device='cpu', activation='tanh', 
                 optimizer_name='adam', learning_rate=0.001, noise_level=0.0,
                 lambda_1_init=0.0, lambda_2_init=0.0):
        
        self.device = torch.device(device)
        
        # Add noise to training data if specified
        if noise_level > 0:
            u = u + noise_level * np.random.normal(0, np.std(u), u.shape)
            v = v + noise_level * np.random.normal(0, np.std(v), v.shape)
            print(f"Added noise with level {noise_level} to training data")
        
        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = torch.tensor(X[:,0:1], dtype=torch.float32, device=self.device)
        self.y = torch.tensor(X[:,1:2], dtype=torch.float32, device=self.device)
        self.t = torch.tensor(X[:,2:3], dtype=torch.float32, device=self.device)
        
        self.u = torch.tensor(u, dtype=torch.float32, device=self.device)
        self.v = torch.tensor(v, dtype=torch.float32, device=self.device)
        
        self.layers = layers
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        
        self.lambda_1_init = lambda_1_init
        self.lambda_2_init = lambda_2_init
        
        # Build neural network
        self.model = NeuralNet(layers, self.lb, self.ub, device, activation).to(self.device)
        
        # Initialize parameters with specified initial values
        self.lambda_1 = torch.nn.Parameter(torch.tensor([lambda_1_init], device=self.device))
        self.lambda_2 = torch.nn.Parameter(torch.tensor([lambda_2_init], device=self.device))
        
        # Create optimizer based on choice
        params = list(self.model.parameters()) + [self.lambda_1, self.lambda_2]
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(params, lr=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Enable automatic differentiation
        self.x.requires_grad = True
        self.y.requires_grad = True
        self.t.requires_grad = True
        
        # Lists to store loss history
        self.loss_history = []
        self.loss_u_history = []
        self.loss_v_history = []
        self.loss_f_u_history = []
        self.loss_f_v_history = []
        self.lambda_1_history = []
        self.lambda_2_history = []
        self.learning_rate_history = []
        
        # Track training time
        self.training_time = 0
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        
        X = torch.cat([x, y, t], 1)
        psi_and_p = self.model(X)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
        
        # First derivatives
        u = torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
        
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
        
        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    def loss_fn(self, x, y, t, u, v):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        
        loss_u = torch.mean(torch.square(u - u_pred))
        loss_v = torch.mean(torch.square(v - v_pred))
        loss_f_u = torch.mean(torch.square(f_u_pred))
        loss_f_v = torch.mean(torch.square(f_v_pred))
        
        return loss_u + loss_v + loss_f_u + loss_f_v, loss_u, loss_v, loss_f_u, loss_f_v
    
    def train(self, nIter):
        start_time = time.time()
        total_start_time = time.time()
        
        # Adam/SGD/RMSprop optimization
        for it in range(nIter):
            self.optimizer.zero_grad()
            
            loss, loss_u, loss_v, loss_f_u, loss_f_v = self.loss_fn(self.x, self.y, self.t, self.u, self.v)
            
            loss.backward()
            self.optimizer.step()
            
            # Store loss values
            self.loss_history.append(loss.item())
            self.loss_u_history.append(loss_u.item())
            self.loss_v_history.append(loss_v.item())
            self.loss_f_u_history.append(loss_f_u.item())
            self.loss_f_v_history.append(loss_f_v.item())
            self.lambda_1_history.append(self.lambda_1.item())
            self.lambda_2_history.append(self.lambda_2.item())
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate_history.append(current_lr)
            
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.item()
                lambda_2_value = self.lambda_2.item()
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, lr: %.2e, Time: %.2f' % 
                      (it, loss.item(), lambda_1_value, lambda_2_value, current_lr, elapsed))
                start_time = time.time()
        
        self.training_time = time.time() - total_start_time
        
        # L-BFGS optimization (only for Adam by default)
        #if self.optimizer_name.lower() == 'adam':
        self.lbfgs_optimize()
    
    def lbfgs_optimize(self):
        """Use scipy's L-BFGS-B optimizer for final optimization"""
        from scipy.optimize import minimize
        
        print('\nStarting L-BFGS-B optimization...')
        start_time = time.time()
        
        # Get initial variables
        trainable_vars = list(self.model.parameters()) + [self.lambda_1, self.lambda_2]
        shapes = [v.shape for v in trainable_vars]
        n_vars = len(shapes)
        
        # Function to flatten variables
        def vars_to_vec(variables):
            return torch.cat([v.flatten() for v in variables], dim=0)
        
        # Function to unflatten variables
        def vec_to_vars(vec):
            variables = []
            idx = 0
            for i in range(n_vars):
                shape = shapes[i]
                size = torch.prod(torch.tensor(shape)).item()
                variables.append(vec[idx:idx+size].reshape(shape))
                idx += size
            return variables
        
        # Define loss function for scipy
        def loss_and_grad(vec):
            # Set variables from vec
            vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
            variables = vec_to_vars(vec_tensor)
            
            for i, param in enumerate(self.model.parameters()):
                param.data.copy_(variables[i])
            self.lambda_1.data.copy_(variables[-2])
            self.lambda_2.data.copy_(variables[-1])
            
            # Clear gradients
            self.model.zero_grad()
            if self.lambda_1.grad is not None:
                self.lambda_1.grad.zero_()
            if self.lambda_2.grad is not None:
                self.lambda_2.grad.zero_()
            
            # Enable gradients for inputs
            self.x.requires_grad_(True)
            self.y.requires_grad_(True)
            self.t.requires_grad_(True)
            
            # Compute loss
            loss, loss_u, loss_v, loss_f_u, loss_f_v = self.loss_fn(self.x, self.y, self.t, self.u, self.v)
            
            # Store loss values for LBFGS iterations
            self.loss_history.append(loss.item())
            self.loss_u_history.append(loss_u.item())
            self.loss_v_history.append(loss_v.item())
            self.loss_f_u_history.append(loss_f_u.item())
            self.loss_f_v_history.append(loss_f_v.item())
            self.lambda_1_history.append(self.lambda_1.item())
            self.lambda_2_history.append(self.lambda_2.item())
            self.learning_rate_history.append(0.0)  # L-BFGS doesn't have a fixed learning rate
            
            # Compute gradients
            loss.backward(retain_graph=True)
            
            # Collect gradients
            grad_list = []
            for param in trainable_vars:
                if param.grad is not None:
                    grad_list.append(param.grad.flatten())
                else:
                    # If no gradient, append zeros
                    grad_list.append(torch.zeros(param.numel(), device=self.device))
            
            grad_vec = torch.cat(grad_list, dim=0)
            
            # Return as float64 for scipy
            return loss.item(), grad_vec.cpu().numpy().astype(np.float64)
        
        # Initial point - convert to float64
        x0 = vars_to_vec(trainable_vars).detach().cpu().numpy().astype(np.float64)
        
        # Callback function
        iteration_count = [0]
        def callback(xk):
            iteration_count[0] += 1
            if iteration_count[0] % 100 == 0:
                vec_tensor = torch.tensor(xk, dtype=torch.float32, device=self.device)
                variables = vec_to_vars(vec_tensor)
                lambda_1_val = variables[-2].item()
                lambda_2_val = variables[-1].item()
                print(f'L-BFGS-B iteration {iteration_count[0]} - l1: {lambda_1_val:.3f}, l2: {lambda_2_val:.5f}')
        
        # Optimize
        res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                      callback=callback, options={'maxiter': 5000})
        
        # Set final variables
        final_vars = vec_to_vars(torch.tensor(res.x, dtype=torch.float32, device=self.device))
        for i, param in enumerate(self.model.parameters()):
            param.data = final_vars[i]
        self.lambda_1.data = final_vars[-2]
        self.lambda_2.data = final_vars[-1]
        
        self.training_time += time.time() - start_time
        print(f'Optimization finished. Final loss: {res.fun:.3e}')
    
    def predict(self, x_star, y_star, t_star):
        x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device, requires_grad=True)
        y_star = torch.tensor(y_star, dtype=torch.float32, device=self.device, requires_grad=True)
        t_star = torch.tensor(t_star, dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.model.eval()
        with torch.no_grad():
            # For prediction, we need to re-enable gradients temporarily
            x_star.requires_grad = True
            y_star.requires_grad = True
            t_star.requires_grad = True
            
            with torch.enable_grad():
                u_star, v_star, p_star, _, _ = self.net_NS(x_star, y_star, t_star)
            
            u_star = u_star.cpu().numpy()
            v_star = v_star.cpu().numpy()
            p_star = p_star.cpu().numpy()
        
        return u_star, v_star, p_star
    
    def save_model(self, filepath):
        """Save the model, parameters, and training history"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lambda_1': self.lambda_1.data,
            'lambda_2': self.lambda_2.data,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'layers': self.layers,
            'lb': self.lb,
            'ub': self.ub,
            'activation': self.activation,
            'optimizer_name': self.optimizer_name,
            'learning_rate': self.learning_rate,
            'noise_level': self.noise_level,
            'lambda_1_init': self.lambda_1_init,
            'lambda_2_init': self.lambda_2_init,
            'loss_history': self.loss_history,
            'loss_u_history': self.loss_u_history,
            'loss_v_history': self.loss_v_history,
            'loss_f_u_history': self.loss_f_u_history,
            'loss_f_v_history': self.loss_f_v_history,
            'lambda_1_history': self.lambda_1_history,
            'lambda_2_history': self.lambda_2_history,
            'learning_rate_history': self.learning_rate_history,
            'training_time': self.training_time
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild the model with saved parameters
        self.model = NeuralNet(checkpoint['layers'], checkpoint['lb'], checkpoint['ub'], 
                               self.device, checkpoint.get('activation', 'tanh')).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load lambda parameters
        self.lambda_1.data = checkpoint['lambda_1']
        self.lambda_2.data = checkpoint['lambda_2']
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        self.loss_history = checkpoint['loss_history']
        self.loss_u_history = checkpoint['loss_u_history']
        self.loss_v_history = checkpoint['loss_v_history']
        self.loss_f_u_history = checkpoint['loss_f_u_history']
        self.loss_f_v_history = checkpoint['loss_f_v_history']
        self.lambda_1_history = checkpoint['lambda_1_history']
        self.lambda_2_history = checkpoint['lambda_2_history']
        self.learning_rate_history = checkpoint.get('learning_rate_history', [])
        self.training_time = checkpoint.get('training_time', 0)
        
        print(f"Model loaded from {filepath}")
    
    def plot_loss_history(self, save_path='loss_history.png'):
        """Plot the loss history and save the figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total loss
        ax1.semilogy(self.loss_history, 'b-')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.set_title(f'Total Loss History ({self.activation}, {self.optimizer_name}, noise={self.noise_level:.2f})')
        ax1.grid(True)
        
        # Component losses
        ax2.semilogy(self.loss_u_history, 'r-', label='Loss u')
        ax2.semilogy(self.loss_v_history, 'g-', label='Loss v')
        ax2.semilogy(self.loss_f_u_history, 'b-', label='Loss f_u')
        ax2.semilogy(self.loss_f_v_history, 'm-', label='Loss f_v')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Component Losses')
        ax2.legend()
        ax2.grid(True)
        
        # Lambda 1 history
        ax3.plot(self.lambda_1_history, 'b-')
        ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='True value')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('lambda_1')
        ax3.set_title(f'lambda_1 History (init={self.lambda_1_init:.2f})')
        ax3.legend()
        ax3.grid(True)
        
        # Lambda 2 history
        ax4.plot(self.lambda_2_history, 'r-')
        ax4.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='True value')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('lambda_2')
        ax4.set_title(f'lambda_2 History (init={self.lambda_2_init:.3f})')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss history plot saved to {save_path}")
        
    def calculate_metrics(self, u_star, v_star, p_star, u_pred, v_pred, p_pred):
        """Calculate comprehensive metrics"""
        error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
        error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
        
        lambda_1_value = self.lambda_1.item()
        lambda_2_value = self.lambda_2.item()
        error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
        error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
        
        # Calculate smoothness metric (gradient magnitude)
        u_grad_mag = np.mean(np.abs(np.gradient(u_pred.flatten())))
        v_grad_mag = np.mean(np.abs(np.gradient(v_pred.flatten())))
        smoothness = (u_grad_mag + v_grad_mag) / 2
        
        metrics = {
            'error_u': error_u,
            'error_v': error_v,
            'error_p': error_p,
            'error_lambda_1': error_lambda_1,
            'error_lambda_2': error_lambda_2,
            'lambda_1': lambda_1_value,
            'lambda_2': lambda_2_value,
            'lambda_1_init': self.lambda_1_init,
            'lambda_2_init': self.lambda_2_init,
            'final_loss': self.loss_history[-1] if self.loss_history else np.inf,
            'training_time': self.training_time,
            'smoothness': smoothness,
            'activation': self.activation,
            'optimizer': self.optimizer_name,
            'noise_level': self.noise_level,
            'iterations': len(self.loss_history)
        }
        
        return metrics


def plot_solution(X_star, u_star, save_path=None, title=''):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(figsize=(8, 6))
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_ablation_study(args):
    """Run ablation study based on user input"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    N_train = args.n_train
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    data = scipy.io.loadmat(args.data_path)
           
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
    
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ablation_results_{args.study_type}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define study parameters
    study_configs = []
    
    if args.study_type == 'activation':
        activations = ['tanh', 'relu', 'sigmoid', 'elu', 'leaky_relu']
        for activation in activations:
            study_configs.append({
                'activation': activation,
                'optimizer': 'adam',
                'noise_level': 0.0,
                'lambda_1_init': 0.0,
                'lambda_2_init': 0.0,
                'name': f'activation_{activation}'
            })
    
    elif args.study_type == 'optimizer':
        optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
        for optimizer in optimizers:
            study_configs.append({
                'activation': 'tanh',
                'optimizer': optimizer,
                'noise_level': 0.0,
                'lambda_1_init': 0.0,
                'lambda_2_init': 0.0,
                'name': f'optimizer_{optimizer}'
            })
    
    elif args.study_type == 'noise':
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        for noise_level in noise_levels:
            study_configs.append({
                'activation': 'tanh',
                'optimizer': 'adam',
                'noise_level': noise_level,
                'lambda_1_init': 0.0,
                'lambda_2_init': 0.0,
                'name': f'noise_{noise_level}'
            })
    
    elif args.study_type == 'lambda_init':
        # Test different initializations for lambda parameters
        # True values: lambda_1 = 1.0, lambda_2 = 0.01
        lambda_1_inits = [0.0, 0.5, 1.0, 2.0, 5.0]
        lambda_2_inits = [0.0, 0.005, 0.01, 0.02, 0.05]
        
        # Test combinations
        for l1_init in lambda_1_inits:
            for l2_init in lambda_2_inits:
                study_configs.append({
                    'activation': 'tanh',
                    'optimizer': 'adam',
                    'noise_level': 0.0,
                    'lambda_1_init': l1_init,
                    'lambda_2_init': l2_init,
                    'name': f'lambda_init_l1_{l1_init}_l2_{l2_init}'
                })
    
    elif args.study_type == 'all':
        # Run a comprehensive study
        activations = ['tanh', 'relu']
        optimizers = ['adam', 'sgd']
        noise_levels = [0.0, 0.1]
        lambda_1_inits = [0.0, 1.0]
        lambda_2_inits = [0.0, 0.01]
        
        for activation in activations:
            for optimizer in optimizers:
                for noise_level in noise_levels:
                    for l1_init in lambda_1_inits:
                        for l2_init in lambda_2_inits:
                            study_configs.append({
                                'activation': activation,
                                'optimizer': optimizer,
                                'noise_level': noise_level,
                                'lambda_1_init': l1_init,
                                'lambda_2_init': l2_init,
                                'name': f'{activation}_{optimizer}_noise{noise_level}_l1_{l1_init}_l2_{l2_init}'
                            })
    
    # Run experiments
    all_metrics = []
    
    for config in study_configs:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*50}")
        
        # Create model
        model = PhysicsInformedNN(
            x_train, y_train, t_train, u_train, v_train, layers, 
            device=device,
            activation=config['activation'],
            optimizer_name=config['optimizer'],
            learning_rate=args.learning_rate,
            noise_level=config['noise_level'],
            lambda_1_init=config.get('lambda_1_init', 0.0),
            lambda_2_init=config.get('lambda_2_init', 0.0)
        )
        
        # Train model
        model.train(args.iterations)
        
        # Save model
        model_path = os.path.join(results_dir, f"model_{config['name']}.pth")
        model.save_model(model_path)
        
        # Plot loss history
        loss_plot_path = os.path.join(results_dir, f"loss_{config['name']}.png")
        model.plot_loss_history(loss_plot_path)
        
        # Prediction
        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        
        # Calculate metrics
        metrics = model.calculate_metrics(u_star, v_star, p_star, u_pred, v_pred, p_pred)
        metrics['config_name'] = config['name']
        all_metrics.append(metrics)
        
        # Plot solutions
        plot_solution(X_star, u_pred, 
                     save_path=os.path.join(results_dir, f"u_pred_{config['name']}.png"),
                     title=f"u prediction - {config['name']}")
        
        plot_solution(X_star, v_pred, 
                     save_path=os.path.join(results_dir, f"v_pred_{config['name']}.png"),
                     title=f"v prediction - {config['name']}")
        
        # Save individual metrics
        metrics_path = os.path.join(results_dir, f"metrics_{config['name']}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics for {config['name']}:")
        print(f"  Error u: {metrics['error_u']:.3e}")
        print(f"  Error v: {metrics['error_v']:.3e}")
        print(f"  Error p: {metrics['error_p']:.3e}")
        print(f"  lambda_1 error: {metrics['error_lambda_1']:.1f}%")
        print(f"  lambda_2 error: {metrics['error_lambda_2']:.1f}%")
        print(f"  Training time: {metrics['training_time']:.1f}s")
        print(f"  Final loss: {metrics['final_loss']:.3e}")
        print(f"  Smoothness: {metrics['smoothness']:.3e}")
    
    # Save all metrics
    all_metrics_path = os.path.join(results_dir, "all_metrics.json")
    with open(all_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Create comparison plots
    create_comparison_plots(all_metrics, results_dir, args.study_type)
    
    print(f"\nAll results saved to: {results_dir}")
    return all_metrics


def create_comparison_plots(metrics, results_dir, study_type):
    """Create comparison plots for ablation studies"""
    
    # Convert to structured format for easier plotting
    data = {}
    for metric in metrics:
        config_name = metric['config_name']
        data[config_name] = metric
    
    # Set up the figure based on study type
    if study_type == 'activation':
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        x_labels = [m['activation'] for m in metrics]
    elif study_type == 'optimizer':
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        x_labels = [m['optimizer'] for m in metrics]
    elif study_type == 'noise':
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        x_labels = [f"{m['noise_level']:.2f}" for m in metrics]
    elif study_type == 'lambda_init':
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, :])
        x_labels = [f"λ1={m['lambda_1_init']:.1f}\nλ2={m['lambda_2_init']:.3f}" for m in metrics]
    else:  # all
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, :])
        x_labels = [m['config_name'] for m in metrics]
    
    # Error comparison
    errors_u = [m['error_u'] for m in metrics]
    errors_v = [m['error_v'] for m in metrics]
    errors_p = [m['error_p'] for m in metrics]
    
    ax1.bar(x_labels, errors_u, label='u error')
    ax1.bar(x_labels, errors_v, bottom=errors_u, label='v error')
    ax1.bar(x_labels, errors_p, bottom=[u+v for u,v in zip(errors_u, errors_v)], label='p error')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('Prediction Errors')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Lambda errors
    lambda1_errors = [m['error_lambda_1'] for m in metrics]
    lambda2_errors = [m['error_lambda_2'] for m in metrics]
    
    x_pos = range(len(x_labels))
    width = 0.35
    ax2.bar([p - width/2 for p in x_pos], lambda1_errors, width, label='lambda_1 error')
    ax2.bar([p + width/2 for p in x_pos], lambda2_errors, width, label='lambda_2 error')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Lambda Parameter Errors')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.legend()
    
    # Training time
    training_times = [m['training_time'] for m in metrics]
    ax3.bar(x_labels, training_times)
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Training Time')
    ax3.tick_params(axis='x', rotation=45)
    
    # Final loss
    final_losses = [m['final_loss'] for m in metrics]
    ax4.bar(x_labels, final_losses)
    ax4.set_ylabel('Loss')
    ax4.set_title('Final Loss')
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', rotation=45)
    
    # Additional plots for comprehensive study or lambda_init study
    if study_type == 'all' or study_type == 'lambda_init':
        # Smoothness
        smoothness = [m['smoothness'] for m in metrics]
        ax5.bar(x_labels, smoothness)
        ax5.set_ylabel('Smoothness')
        ax5.set_title('Solution Smoothness')
        ax5.tick_params(axis='x', rotation=45)
        
        # Lambda values
        lambda1_vals = [m['lambda_1'] for m in metrics]
        lambda2_vals = [m['lambda_2'] for m in metrics]
        
        ax6.plot(x_labels, lambda1_vals, 'o-', label='lambda_1')
        ax6.plot(x_labels, lambda2_vals, 's-', label='lambda_2')
        ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='lambda_1 true')
        ax6.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='lambda_2 true')
        ax6.set_ylabel('Value')
        ax6.set_title('Learned Lambda Values')
        ax6.legend()
        ax6.tick_params(axis='x', rotation=45)
        
        if study_type == 'lambda_init':
            # Create heatmap for lambda initialization study
            import numpy as np
            
            # Get unique lambda init values
            unique_lambda1 = sorted(set(m['lambda_1_init'] for m in metrics))
            unique_lambda2 = sorted(set(m['lambda_2_init'] for m in metrics))
            
            # Create error heatmap
            heatmap_data = []
            for l1 in unique_lambda1:
                row = []
                for l2 in unique_lambda2:
                    # Find matching configuration
                    matching = [m for m in metrics 
                               if m['lambda_1_init'] == l1 
                               and m['lambda_2_init'] == l2]
                    if matching:
                        row.append(matching[0]['error_u'])
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            im = ax7.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
            ax7.set_xticks(range(len(unique_lambda2)))
            ax7.set_xticklabels([f'{l:.3f}' for l in unique_lambda2])
            ax7.set_yticks(range(len(unique_lambda1)))
            ax7.set_yticklabels([f'{l:.1f}' for l in unique_lambda1])
            ax7.set_xlabel('lambda_2 Initial Value')
            ax7.set_ylabel('lambda_1 Initial Value')
            ax7.set_title('Error Heatmap (u component) - Lambda Initialization Sensitivity')
            
            # Add true values as markers
            l1_true_idx = np.argmin(np.abs(np.array(unique_lambda1) - 1.0))
            l2_true_idx = np.argmin(np.abs(np.array(unique_lambda2) - 0.01))
            ax7.scatter(l2_true_idx, l1_true_idx, marker='*', s=500, c='red', 
                       edgecolors='white', linewidth=2, label='True values')
            ax7.legend()
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax7)
            cbar.set_label('Relative Error')
            
            # Add text annotations for specific values
            for i in range(len(unique_lambda1)):
                for j in range(len(unique_lambda2)):
                    if not np.isnan(heatmap_data[i][j]) and heatmap_data[i][j] < 0.1:
                        text = ax7.text(j, i, f'{heatmap_data[i][j]:.3f}',
                                       ha="center", va="center", color="white", fontsize=8)
        else:
            # Create heatmap of errors for other studies
            import seaborn as sns
            
            # Organize data for heatmap
            heatmap_data = []
            unique_activations = sorted(set(m['activation'] for m in metrics))
            unique_optimizers = sorted(set(m['optimizer'] for m in metrics))
            unique_noise = sorted(set(m['noise_level'] for m in metrics))
            
            for act in unique_activations:
                for opt in unique_optimizers:
                    row = []
                    for noise in unique_noise:
                        # Find matching configuration
                        matching = [m for m in metrics 
                                   if m['activation'] == act 
                                   and m['optimizer'] == opt 
                                   and m['noise_level'] == noise]
                        if matching:
                            row.append(matching[0]['error_u'])
                        else:
                            row.append(np.nan)
                    heatmap_data.append(row)
            
            im = ax7.imshow(heatmap_data, aspect='auto', cmap='viridis')
            ax7.set_xticks(range(len(unique_noise)))
            ax7.set_xticklabels([f'{n:.2f}' for n in unique_noise])
            ax7.set_yticks(range(len(unique_activations) * len(unique_optimizers)))
            ax7.set_yticklabels([f'{act}-{opt}' for act in unique_activations for opt in unique_optimizers])
            ax7.set_xlabel('Noise Level')
            ax7.set_ylabel('Activation-Optimizer')
            ax7.set_title('Error Heatmap (u component)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax7)
            cbar.set_label('Relative Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_{study_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create convergence comparison plot
    plot_convergence_comparison(metrics, results_dir, study_type)


def plot_convergence_comparison(metrics, results_dir, study_type):
    """Plot convergence curves for different configurations"""
    
    plt.figure(figsize=(12, 8))
    
    # Load and plot loss histories for each configuration
    for metric in metrics:
        config_name = metric['config_name']
        model_path = os.path.join(results_dir, f"model_{config_name}.pth")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            loss_history = checkpoint['loss_history']
            
            # Subsample for cleaner plot if needed
            if len(loss_history) > 1000:
                indices = np.linspace(0, len(loss_history)-1, 1000, dtype=int)
                loss_history = [loss_history[i] for i in indices]
            
            plt.semilogy(loss_history, label=config_name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Convergence Comparison - {study_type.capitalize()} Study')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'convergence_{study_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional plot for lambda initialization study
    if study_type == 'lambda_init':
        create_lambda_trajectory_plot(metrics, results_dir)


def create_lambda_trajectory_plot(metrics, results_dir):
    """Create a plot showing lambda parameter trajectories for different initializations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get unique initial values
    unique_configs = []
    for metric in metrics:
        config = (metric['lambda_1_init'], metric['lambda_2_init'])
        if config not in unique_configs:
            unique_configs.append(config)
    
    # Plot lambda_1 trajectories
    for l1_init, l2_init in unique_configs:
        # Find matching configuration
        matching_metric = [m for m in metrics 
                          if m['lambda_1_init'] == l1_init 
                          and m['lambda_2_init'] == l2_init][0]
        
        config_name = matching_metric['config_name']
        model_path = os.path.join(results_dir, f"model_{config_name}.pth")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            lambda_1_history = checkpoint['lambda_1_history']
            
            # Subsample for cleaner plot
            if len(lambda_1_history) > 1000:
                indices = np.linspace(0, len(lambda_1_history)-1, 1000, dtype=int)
                lambda_1_history = [lambda_1_history[i] for i in indices]
                x_axis = indices
            else:
                x_axis = range(len(lambda_1_history))
            
            ax1.plot(x_axis, lambda_1_history, 
                    label=f'init: lambda_1={l1_init:.1f}, lambda_2={l2_init:.3f}', 
                    linewidth=2)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='True value')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('lambda_1')
    ax1.set_title('lambda_1 Convergence from Different Initializations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot lambda_2 trajectories
    for l1_init, l2_init in unique_configs:
        # Find matching configuration
        matching_metric = [m for m in metrics 
                          if m['lambda_1_init'] == l1_init 
                          and m['lambda_2_init'] == l2_init][0]
        
        config_name = matching_metric['config_name']
        model_path = os.path.join(results_dir, f"model_{config_name}.pth")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            lambda_2_history = checkpoint['lambda_2_history']
            
            # Subsample for cleaner plot
            if len(lambda_2_history) > 1000:
                indices = np.linspace(0, len(lambda_2_history)-1, 1000, dtype=int)
                lambda_2_history = [lambda_2_history[i] for i in indices]
                x_axis = indices
            else:
                x_axis = range(len(lambda_2_history))
            
            ax2.plot(x_axis, lambda_2_history, 
                    label=f'init: lambda_1={l1_init:.1f}, lambda_2={l2_init:.3f}', 
                    linewidth=2)
    
    ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='True value')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('lambda_2')
    ax2.set_title('lambda_2 Convergence from Different Initializations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lambda_trajectories.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN Ablation Study')
    parser.add_argument('--study_type', type=str, default='activation',
                        choices=['activation', 'optimizer', 'noise', 'lambda_init', 'all'],
                        help='Type of ablation study to run')
    parser.add_argument('--n_train', type=int, default=5000,
                        help='Number of training points')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_path', type=str, default='../Data/cylinder_nektar_wake.mat',
                        help='Path to data file')
    parser.add_argument('--run_single', action='store_true',
                        help='Run single configuration instead of ablation study')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function for single run')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for single run')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Noise level for single run')
    parser.add_argument('--lambda_1_init', type=float, default=0.0,
                        help='Initial value for lambda_1 for single run')
    parser.add_argument('--lambda_2_init', type=float, default=0.0,
                        help='Initial value for lambda_2 for single run')
    
    args = parser.parse_args()
    
    if args.run_single:
        # Run single configuration
        print("Running single configuration...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load data
        data = scipy.io.loadmat(args.data_path)
        U_star = data['U_star']
        P_star = data['p_star']
        t_star = data['t']
        X_star = data['X_star']
        
        N = X_star.shape[0]
        T = t_star.shape[0]
        
        # Prepare data
        XX = np.tile(X_star[:,0:1], (1,T))
        YY = np.tile(X_star[:,1:2], (1,T))
        TT = np.tile(t_star, (1,N)).T
        
        UU = U_star[:,0,:]
        VV = U_star[:,1,:]
        PP = P_star
        
        x = XX.flatten()[:,None]
        y = YY.flatten()[:,None]
        t = TT.flatten()[:,None]
        
        u = UU.flatten()[:,None]
        v = VV.flatten()[:,None]
        p = PP.flatten()[:,None]
        
        # Training data
        idx = np.random.choice(N*T, args.n_train, replace=False)
        x_train = x[idx,:]
        y_train = y[idx,:]
        t_train = t[idx,:]
        u_train = u[idx,:]
        v_train = v[idx,:]
        
        # Create and train model
        layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
        model = PhysicsInformedNN(
            x_train, y_train, t_train, u_train, v_train, layers,
            device=device,
            activation=args.activation,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            noise_level=args.noise_level,
            lambda_1_init=args.lambda_1_init,
            lambda_2_init=args.lambda_2_init
        )
        
        model.train(args.iterations)
        model.save_model('single_run_model.pth')
        model.plot_loss_history('single_run_loss.png')
        
        # Test
        snap = np.array([100])
        x_star = X_star[:,0:1]
        y_star = X_star[:,1:2]
        t_star = TT[:,snap]
        
        u_star = U_star[:,0,snap]
        v_star = U_star[:,1,snap]
        p_star = P_star[:,snap]
        
        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        metrics = model.calculate_metrics(u_star, v_star, p_star, u_pred, v_pred, p_pred)
        
        print("\nResults:")
        print(f"Error u: {metrics['error_u']:.3e}")
        print(f"Error v: {metrics['error_v']:.3e}")
        print(f"Error p: {metrics['error_p']:.3e}")
        print(f"lambda_1 error: {metrics['error_lambda_1']:.1f}%")
        print(f"lambda_2 error: {metrics['error_lambda_2']:.1f}%")
        
    else:
        # Run ablation study
        run_ablation_study(args)
        
        
##Initialization setup
#python3 NavierStokes_ablation.py --study_type lambda_init --iterations 60000

##Run activation function study
#python3 NavierStokes_ablation.py --study_type activation --iterations 60000

##Run optimizer study:
#python3 NavierStokes_ablation.py --study_type optimizer --iterations 60000

##Run noise robustness study
#python3 NavierStokes_ablation.py --study_type noise --iterations 60000


