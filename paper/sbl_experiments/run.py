# %% Imports
from jax import numpy as jnp, random
from modax.data.kdv import doublesoliton
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim
import jax
from modax.training import train_max_iter
from modax.training.losses.SBL import loss_fn_SBL

script_dir = "/home/gert-jan/Documents/modax/paper/sbl_experiments/runs/"
key = random.PRNGKey(42)
n_runs = 1
max_iterations = 10000


# Running noise levels
x = jnp.linspace(-10, 10, 100)
t = jnp.linspace(0.1, 1.0, 10)
t_grid, x_grid = jnp.meshgrid(t, x, idexing="ij")
u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)

# Defning model and optimizers
model = Deepmod([30, 30, 30, 1], (4, 3))
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
noise_levels = jnp.arange(0.1, 1.01, 0.10)
# noise_levels = jax.ops.index_update(
#    noise_levels, 0, 0.01
# )  # setting first level to 0.01

for noise in noise_levels:
    y_noisy = y + noise * jnp.std(y) * random.normal(key, y.shape)
    update_fn = create_update(loss_fn_SBL, (model, X, y_noisy, True))
    for run_idx, subkey in enumerate(random.split(key, n_runs)):
        print(f"Starting noise run {run_idx}")
        variables = model.init(subkey, X)
        state, params = variables.pop("params")
        state = (state, {"prior_init": None})  # adding prior to state
        optimizer = optimizer_def.create(params)
        train_max_iter(
            update_fn,
            optimizer,
            state,
            max_iterations,
            log_dir=script_dir + f"noise_{noise:.2f}_run_{run_idx}/",
        )
# RUnning different sparsity.
# Defning model and optimizers
model = Deepmod([30, 30, 30, 1], (4, 3))
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
for n_x in jnp.arange(10, 101, 10):
    x = jnp.linspace(-10, 10, n_x)
    t = jnp.linspace(0.1, 1.0, 10)
    t_grid, x_grid = jnp.meshgrid(t, x, idexing="ij")
    u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])

    X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
    y = u.reshape(-1, 1)
    y_noisy = y + 0.10 * jnp.std(y) * random.normal(key, y.shape)

    update_fn = create_update(loss_fn_SBL, (model, X, y_noisy, True))
    for run_idx, subkey in enumerate(random.split(key, n_runs)):
        print(f"Starting sparsity run {run_idx}")
        variables = model.init(subkey, X)
        state, params = variables.pop("params")
        state = (state, {"prior_init": None})  # adding prior to state
        optimizer = optimizer_def.create(params)
        train_max_iter(
            update_fn,
            optimizer,
            state,
            max_iterations,
            log_dir=script_dir + f"nx_{n_x}_run_{run_idx}/",
        )

