from modax.data.burgers import burgers
from modax.logging import Logger

from flax import optim

import jax
from jax import random, numpy as jnp

from code import loss_fn_SBL, DeepmodSBL, create_update


# Making dataset
x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.5, 5.0, 20)
key = random.PRNGKey(42)

t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X_train = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y_train = u.reshape(-1, 1)
y_train += 0.01 * jnp.std(y_train) * jax.random.normal(key, y_train.shape)

n_samples = X_train.shape[0]
hyper_prior = jnp.stack([n_samples / 2, 1 / (n_samples / 2 * 1e-4)], axis=0)
model = DeepmodSBL([50, 50, 1], hyper_prior, tol=1e-5)

"""
variables = model.init(key, X_train)


optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)
# Compiling train step
update = create_update(loss_fn_SBL, model=model, x=X_train, y=y_train)
_ = update(optimizer, state)  # triggering compilation
# Running to convergence
max_epochs = 10000
logger = Logger()
for epoch in jnp.arange(max_epochs):
    (optimizer, state), metrics = update(optimizer, state)
    if epoch % 1000 == 0:
        print(f"Loss step {epoch}: {metrics['loss']}")
    if epoch % 25 == 0:
        logger.write(metrics, epoch)
logger.close()
"""

