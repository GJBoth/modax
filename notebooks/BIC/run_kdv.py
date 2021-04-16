# %% Imports
from jax import numpy as jnp, random
from modax.data.kdv import doublesoliton
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training import train_max_iter
from BIC import loss_fn_SBL

script_dir = "/home/gert-jan/Documents/modax/notebooks/BIC/runs/"
key = random.PRNGKey(42)
noise = 0.50
max_iterations = 100000

# Making data
x = jnp.linspace(-10, 10, 100)
t = jnp.linspace(0.1, 1.0, 10)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += 0.50 * jnp.std(y) * random.normal(key, y.shape)

# Defning model and optimizers
model = Deepmod([30, 30, 30, 1])
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)

# Running warm restart bayes
update_fn = create_update(loss_fn_SBL, (model, X, y, True))
variables = model.init(key, X)
state, params = variables.pop("params")
state = (state, {"prior_init": None})  # adding prior to state
optimizer = optimizer_def.create(params)
train_max_iter(
    update_fn,
    optimizer,
    state,
    max_iterations,
    log_dir=script_dir + f"BIC_parallel_kdv_50/",
)
