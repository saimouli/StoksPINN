import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers):
        
        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        
        self.u = u
        self.v = v
        
        self.layers = layers
        
        # Build neural network
        self.model = self.build_model()
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_lbfgs = tf.keras.optimizers.Adam(learning_rate=0.001)  # Will use scipy for L-BFGS-B later
        
    def build_model(self):
        # Build neural network using Keras functional API
        inputs = tf.keras.Input(shape=(3,))
        
        # Normalize inputs
        H = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        
        # Hidden layers
        for units in self.layers[1:-1]:
            H = tf.keras.layers.Dense(units, activation='tanh',
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())(H)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.layers[-1], 
                                      kernel_initializer=tf.keras.initializers.GlorotNormal())(H)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    @tf.function
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            
            X = tf.concat([x, y, t], 1)
            psi_and_p = self.model(X)
            psi = psi_and_p[:, 0:1]
            p = psi_and_p[:, 1:2]
            
            u = tape.gradient(psi, y)
            v = -tape.gradient(psi, x)
            
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(y)
            tape2.watch(t)
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                tape1.watch(y)
                tape1.watch(t)
                
                X = tf.concat([x, y, t], 1)
                psi_and_p = self.model(X)
                psi = psi_and_p[:, 0:1]
                p = psi_and_p[:, 1:2]
                
                u = tape1.gradient(psi, y)
                v = -tape1.gradient(psi, x)
                
            u_t = tape2.gradient(u, t)
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            
            v_t = tape2.gradient(v, t)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)
            
        # Second derivatives
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(x)
            tape3.watch(y)
            
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                tape2.watch(t)
                
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(x)
                    tape1.watch(y)
                    tape1.watch(t)
                    
                    X = tf.concat([x, y, t], 1)
                    psi_and_p = self.model(X)
                    psi = psi_and_p[:, 0:1]
                    
                    u = tape1.gradient(psi, y)
                    v = -tape1.gradient(psi, x)
                    
                u_x = tape2.gradient(u, x)
                u_y = tape2.gradient(u, y)
                v_x = tape2.gradient(v, x)
                v_y = tape2.gradient(v, y)
                
            u_xx = tape3.gradient(u_x, x)
            u_yy = tape3.gradient(u_y, y)
            v_xx = tape3.gradient(v_x, x)
            v_yy = tape3.gradient(v_y, y)
            
        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    @tf.function
    def loss_fn(self, x, y, t, u, v):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        
        loss_u = tf.reduce_mean(tf.square(u - u_pred))
        loss_v = tf.reduce_mean(tf.square(v - v_pred))
        loss_f_u = tf.reduce_mean(tf.square(f_u_pred))
        loss_f_v = tf.reduce_mean(tf.square(f_v_pred))
        
        return loss_u + loss_v + loss_f_u + loss_f_v
    
    @tf.function
    def train_step(self, x, y, t, u, v):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, y, t, u, v)
        
        trainable_vars = self.model.trainable_variables + [self.lambda_1, self.lambda_2]
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def train(self, nIter):
        # Convert to TensorFlow tensors
        x = tf.constant(self.x, dtype=tf.float32)
        y = tf.constant(self.y, dtype=tf.float32)
        t = tf.constant(self.t, dtype=tf.float32)
        u = tf.constant(self.u, dtype=tf.float32)
        v = tf.constant(self.v, dtype=tf.float32)
        
        start_time = time.time()
        
        # Adam optimization
        for it in range(nIter):
            loss_value = self.train_step(x, y, t, u, v)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.numpy()[0]
                lambda_2_value = self.lambda_2.numpy()[0]
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
        
        # L-BFGS-B optimization using scipy
        self.scipy_optimize(x, y, t, u, v)
    
    def scipy_optimize(self, x, y, t, u, v):
        """Use scipy's L-BFGS-B optimizer for final optimization"""
        from scipy.optimize import minimize
        
        # Get initial variables
        trainable_vars = self.model.trainable_variables + [self.lambda_1, self.lambda_2]
        shapes = [v.shape for v in trainable_vars]
        n_vars = len(shapes)
        
        # Function to flatten variables
        def vars_to_vec(variables):
            return tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)
        
        # Function to unflatten variables
        def vec_to_vars(vec):
            variables = []
            idx = 0
            for i in range(n_vars):
                shape = shapes[i]
                size = tf.reduce_prod(shape)
                variables.append(tf.reshape(vec[idx:idx+size], shape))
                idx += size
            return variables
        
        # Define loss function for scipy
        def loss_and_grad(vec):
            variables = vec_to_vars(tf.constant(vec, dtype=tf.float32))
            
            # Set variables
            for i in range(len(self.model.trainable_variables)):
                self.model.trainable_variables[i].assign(variables[i])
            self.lambda_1.assign(variables[-2])
            self.lambda_2.assign(variables[-1])
            
            with tf.GradientTape() as tape:
                loss = self.loss_fn(x, y, t, u, v)
            
            gradients = tape.gradient(loss, trainable_vars)
            grad_vec = vars_to_vec(gradients)
            
            # Return as float64 for scipy
            return loss.numpy().astype(np.float64), grad_vec.numpy().astype(np.float64)
        
        # Initial point - convert to float64
        x0 = vars_to_vec(trainable_vars).numpy().astype(np.float64)
        
        # Callback function
        def callback(xk):
            variables = vec_to_vars(tf.constant(xk, dtype=tf.float32))
            lambda_1_val = variables[-2].numpy()[0]
            lambda_2_val = variables[-1].numpy()[0]
            print(f'L-BFGS-B - l1: {lambda_1_val:.3f}, l2: {lambda_2_val:.5f}')
        
        # Optimize
        print('\nStarting L-BFGS-B optimization...')
        res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                      callback=callback, options={'maxiter': 5000}) #50000
        
        # Set final variables
        final_vars = vec_to_vars(tf.constant(res.x, dtype=tf.float32))
        for i in range(len(self.model.trainable_variables)):
            self.model.trainable_variables[i].assign(final_vars[i])
        self.lambda_1.assign(final_vars[-2])
        self.lambda_2.assign(final_vars[-1])
        
        print(f'Optimization finished. Final loss: {res.fun:.3e}')
    
    def predict(self, x_star, y_star, t_star):
        x_star = tf.constant(x_star, dtype=tf.float32)
        y_star = tf.constant(y_star, dtype=tf.float32)
        t_star = tf.constant(t_star, dtype=tf.float32)
        
        u_star, v_star, p_star, _, _ = self.net_NS(x_star, y_star, t_star)
        
        return u_star.numpy(), v_star.numpy(), p_star.numpy()

def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
      
    N_train = 5000
    
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
           
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
    
    ######################################################################
    ######################## Noiseless Data ##############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(50000) #200000
    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.lambda_1.numpy()[0]
    lambda_2_value = model.lambda_2.numpy()[0]
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results
#    plot_solution(X_star, u_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    # noise = 0.01        
    # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    # v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

    # # Training
    # model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    # model.train(200000)
        
    # lambda_1_value_noisy = model.lambda_1.numpy()[0]
    # lambda_2_value_noisy = model.lambda_2.numpy()[0]
      
    # error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    # error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100
        
    # print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    # print('Error l2: %.5f%%' % (error_lambda_2_noisy))     

             
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
     # Load Data
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
           
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    modes = data_vort['modes'].item() # Replaced np.asscalar
    nel = data_vort['nel'].item()     # Replaced np.asscalar
    
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: Vorticity ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    
    
    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################        
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0],  projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    ########      v(t,x,y)     ###################        
    ax = plt.subplot(gs1[:, 1],  projection='3d')
    ax.axis('off')
    
    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    #make dir if not exists
    import os
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    savefig('./figures/NavierStokes_data') 

    
    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')
    
    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)
    
    
    ######## Row 3: Table #######################
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])
    ax.axis('off')
    
    s = r'$\begin{tabular}{|c|c|}';
    s = s + r' \hline'
    s = s + r' Correct PDE & $\begin{array}{c}'
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \end{array}$ \\ '
    # s = s + r' \hline'
    # s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    # s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    # s = s + r' \\'
    # s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    # s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' \end{tabular}$'
 
    ax.text(0.015,0.0,s)
    
    savefig('./figures/NavierStokes_prediction') 
