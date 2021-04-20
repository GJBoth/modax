# %% Imports
from jax import numpy as jnp, random
import jax
from modax.data.burgers import burgers
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training import train_max_iter
from modax.training.losses.standard import loss_fn_pinn
from modax.training.losses.multitask import (
    loss_fn_multitask_precalc,
    loss_fn_pinn_bayes_mse_hyperprior,
)

script_dir = "/home/gert-jan/Documents/modax/paper/multitask/runs/"
key = random.PRNGKey(42)
noise = 0.10
n_runs = 5
max_iterations = 5000

# Making data
x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.1, 5.0, 2)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += noise * jnp.std(y) * random.normal(key, y.shape)


# Defning model and optimizers
model = Deepmod([30, 30, 30, 1])
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)

# Running multitask
update_fn = create_update(loss_fn_multitask_precalc, (model, X, y))
for run_idx, subkey in enumerate(random.split(key, n_runs)):
    print(f"Starting multitask run {run_idx}")
    variables = model.init(subkey, X)
    state, params = variables.pop("params")
    optimizer = optimizer_def.create(params)
    train_max_iter(
        update_fn,
        optimizer,
        state,
        max_iterations,
        log_dir=script_dir + f"multitask_run_{run_idx}/",
    )

# Running bayesian multitask
update_fn = create_update(loss_fn_pinn_bayes_mse_hyperprior, (model, X, y))
for run_idx, subkey in enumerate(random.split(key, n_runs)):
    print(f"Starting bayes run {run_idx}")
    variables = model.init(subkey, X)
    state, params = variables.pop("params")
    optimizer = optimizer_def.create(params)
    train_max_iter(
        update_fn,
        optimizer,
        state,
        max_iterations,
        log_dir=script_dir + f"bayes_run_{run_idx}/",
    )

# Running normal PINN
update_fn = create_update(loss_fn_pinn, (model, X, y))
for run_idx, subkey in enumerate(random.split(key, n_runs)):
    print(f"Starting pinn run {run_idx}")
    variables = model.init(subkey, X)
    state, params = variables.pop("params")
    optimizer = optimizer_def.create(params)
    train_max_iter(
        update_fn,
        optimizer,
        state,
        max_iterations,
        log_dir=script_dir + f"pinn_run_{run_idx}/",
    )
