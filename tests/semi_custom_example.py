# %% Imports
from jax import random, numpy as jnp

from modax.data.burgers import burgers
from modax.losses.standard import loss_fn_pinn_stateful
from modax.models import Deepmod
from modax.training import train
from modax.linear_model.mask_estimator import ThresholdedLasso

from flax import optim


# %% Making data
key = random.PRNGKey(42)

x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.5, 5.0, 20)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += 0.10 * jnp.std(y) * random.normal(key, y.shape)

# %%

model = Deepmod([30, 30, 30, 1])
variables = model.init(key, X)

optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)

mask_fn = ThresholdedLasso(threshold=0.1)

optimizer, state = train(
    model,
    optimizer,
    state,
    loss_fn_pinn_stateful,
    mask_fn,
    X,
    y,
    max_epochs=1e4,
    split=0.8,
    rand_seed=42,
)

