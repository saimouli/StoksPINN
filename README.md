# StoksPINN
This code implements a PINN to learn the solution of the 2D incompressible Navier-Stokes equations (for fluid dynamics) from sparse data, using both data and physics. The physics is incorporated as soft constraint during the training of a neural network.

## Installations 
For GPU
```
pip install tensorflow[and-cuda]
```

## To Run
Navigate to main/continous_time_identification (Navier-Stokes)
```
python3 NavierStokes.py
```

# Created a pytorch version to study ablations
# initialization setup
```
python3 NavierStokes_ablation.py --study_type lambda_init --iterations 60000
```

# Run activation function study
```
python3 NavierStokes_ablation.py --study_type activation --iterations 60000
```
# Run optimizer study:
```
python3 NavierStokes_ablation.py --study_type optimizer --iterations 60000
```
# Run noise robustness study
```
python3 NavierStokes_ablation.py --study_type noise --iterations 60000
```
For more information, please refer to the following: (https://maziarraissi.github.io/PINNs/)

  - Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).

  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).
    
Base code is from 
https://github.com/Shengfeng233/PINN-for-NS-equation.git
https://github.com/chen-yingfa/pinn-torch
