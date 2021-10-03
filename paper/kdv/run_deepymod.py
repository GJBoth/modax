# %% Imports
from jax import numpy as jnp, random
from modax.data.kdv import doublesoliton

# General imports
import numpy as np
import torch

# DeepMoD functions
from deepymod import DeepMoD
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)


script_dir = "/home/gert-jan/Documents/modax/paper/kdv/runs_rewrite/"
key = random.PRNGKey(0)
noise = 0.20
n_runs = 5
max_iterations = 20000

# Making data
x = jnp.linspace(-6, 7, 50)
t = jnp.linspace(0.1, 3.0, 40)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[-5.0, 0.0])

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += noise * jnp.std(y) * random.normal(key, y.shape)

for run_idx in np.arange(n_runs):
    # Running deepymod
    network = NN(2, [30, 30, 30], 1)
    library = Library1D(poly_order=3, diff_order=4)
    estimator = Threshold(0.1)
    sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
    constraint = LeastSquares()
    model = DeepMoD(network, library, estimator, constraint)

    # Defining optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=2e-3
    )
    train(
        model,
        torch.from_numpy(np.array(X)),
        torch.from_numpy(np.array(y)),
        optimizer,
        sparsity_scheduler,
        log_dir=f"paper/kdv/runs_rewrite/deepymod_{run_idx}/",
        split=0.8,
        max_iterations=5000,
        patience=5000,
    )
