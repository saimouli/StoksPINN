# StoksPINN

## Installations 
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

## To Run
```
python3 train.py
python3 plot.py
```
## Continuous Inverse Navier-Stokes Equation
Given the 2D nonlinear Navier-Stokes equation:

$$u_t + \lambda_{1}(uu_x + vu_y) = -p_x + \lambda_{2}(u_{xx} + u_{yy}), v_t + \lambda_{1}(uv_x + vv_y) = -p_y + \lambda_{2}(v_{xx} + v_{yy}),$$

where $u(t, x, y)$ and $v(t, x, y)$ are the x and y components of the velocity field, and $p(t, x, y)$ is the pressure, we seek the unknowns $\lambda = (\lambda_1, \lambda_2)$. When required, we integrate the constraints:

$$ 0 = u_x + v_y, u = \psi_y, v = -\psi_x,$$

We use a dual-output neural network to approximate $[\psi(t, x, y), p(t, x, y)]$, leading to a physics-informed neural network $[f(t, x, y), g(t, x, y)]$. 


The base code of the repository is from: https://github.com/Shengfeng233/PINN-for-NS-equation.git
