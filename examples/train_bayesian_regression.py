# %% Imports
from jax import numpy as jnp, random

from modax.data.burgers import burgers
from modax.data.kdv import doublesoliton
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training.losses.bayesian_regression import loss_fn_bayesian_ridge
from modax.training import train_max_iter

# %% Making data; Burgers or KdV
dataset = "kdv"
key = random.PRNGKey(42)

if dataset == "burgers":
    x = jnp.linspace(-3, 4, 50)
    t = jnp.linspace(0.5, 5.0, 20)
    t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
    u = burgers(x_grid, t_grid, 0.1, 1.0)

    X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
    y = u.reshape(-1, 1)
    y += 0.10 * jnp.std(y) * random.normal(key, y.shape)

elif dataset == "kdv":
    key = random.PRNGKey(42)
    x = jnp.linspace(-10, 10, 100)
    t = jnp.linspace(0.1, 1.0, 10)
    t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
    u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])

    X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
    y = u.reshape(-1, 1)
    y += 0.10 * jnp.std(y) * random.normal(key, y.shape)
else:
    raise NotImplementedError

# %% Building model and params
model = Deepmod([30, 30, 30, 1])
variables = model.init(key, X)

optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)

state = (state, {"prior_init": None})  # adding prior to state
update_fn = create_update(loss_fn_bayesian_ridge, (model, X, y, True))

optimizer, state = train_max_iter(update_fn, optimizer, state, 5000)
