# %% Imports
import jax
from jax import random, numpy as jnp
from flax import optim
from modax.networks import MLP
from modax.training import train_step_pinn
from modax.feature_generators import library_backward
from modax.data.burgers import burgers
from modax.layers import LeastSquares
from time import time

# Making dataset
x = jnp.linspace(-3, 4, 100)
t = jnp.linspace(0.5, 5.0, 20)

t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X_train = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y_train = u.reshape(-1, 1)

# Instantiating model and optimizers
model = MLP(features=[50, 50, 1])
constraint = LeastSquares()
key = random.PRNGKey(42)
params = model.init(key, X_train[[0], :])
optimizer = optim.Adam(learning_rate=1e-3)
optimizer = optimizer.create(params)

# Compiling train step
step = train_step_pinn(model, library_backward, constraint, X_train, y_train)
_ = step(optimizer)  # triggering compilation

# Running to convergence
max_epochs = 10001
t_start = time()
for i in jnp.arange(max_epochs):
    optimizer, loss = step(optimizer)
    if i % 1000 == 0:
        print(f"Loss step {i}: {loss}")
t_end = time()
print(t_end - t_start)
