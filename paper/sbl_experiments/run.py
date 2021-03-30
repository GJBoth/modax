# %% Imports
from jax import numpy as jnp, random
from modax.data.burgers import burgers
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training import train_max_iter
from modax.training.losses.SBL import loss_fn_SBL

script_dir = "/home/gert-jan/Documents/modax/paper/sbl_experiments/runs/"
key = random.PRNGKey(42)
n_runs = 1
max_iterations = 5000

# Running noise levels
x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.5, 5.0, 20)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)

# Defning model and optimizers
model = Deepmod([30, 30, 30, 1])
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
noise_levels = jnp.arange(0, 1.0, 0.05)
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
