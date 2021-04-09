# %% Imports
from jax import numpy as jnp, random
from modax.data.kdv import doublesoliton
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training import train_max_iter
from modax.training.losses.standard import loss_fn_pinn
from modax.training.losses.bayesian_regression import loss_fn_bayesian_ridge

script_dir = (
    "/home/gert-jan/Documents/modax/paper/bayesian_regression_experiments/runs/"
)
key = random.PRNGKey(42)
noise = 0.10
n_runs = 1
max_iterations = 5000

## Burgers

# Making data
x = jnp.linspace(-10, 10, 100)
t = jnp.linspace(0.1, 1.0, 10)
t_grid, x_grid = jnp.meshgrid(t, x, idexing="ij")
u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])


X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += noise * jnp.std(y) * random.normal(key, y.shape)


# Defning model and optimizers
model = Deepmod([30, 30, 30, 1])
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)

# Running PINN with overcomplete library
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
        log_dir=script_dir + f"kdv_pinn_run_{run_idx}/",
    )

# Running bayes with overcomplete library
update_fn = create_update(loss_fn_bayesian_ridge, (model, X, y, True))
for run_idx, subkey in enumerate(random.split(key, n_runs)):
    print(f"Starting bayes run {run_idx}")
    variables = model.init(subkey, X)
    state, params = variables.pop("params")
    state = (state, {"prior_init": None})  # adding prior to state
    optimizer = optimizer_def.create(params)
    train_max_iter(
        update_fn,
        optimizer,
        state,
        max_iterations,
        log_dir=script_dir + f"kdv_bayes_run_{run_idx}/",
    )

