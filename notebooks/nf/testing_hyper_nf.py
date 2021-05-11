# %%
from functools import partial
import jax.numpy as jnp
import jax.random as random
from flax import optim
from modax.training import create_update, train_max_iter
from nf import AmortizedNormalizingFlow
from data import random_walk


key = random.PRNGKey(42)
# %% Making sample dataset


def dataset():
    n_steps = 100
    n_walkers = 50
    D = 0.5
    dt = 0.05

    def initial_condition(loc, width, key, shape):
        return loc + width * random.normal(key, shape)
    ini = partial(initial_condition, 1, 2.0)
    locs, t = random_walk(key, ini, n_steps, n_walkers, D, dt=dt)
    return jnp.expand_dims(locs, axis=-1), jnp.expand_dims(t, axis=-1)


def loss_fn(params, state, model, x):
    log_p = model.apply(params, x)
    loss = -jnp.sum(log_p)
    metrics = {"loss": loss}
    return loss, (state, metrics, log_p)


X = dataset()
# %% Initializing model
model = AmortizedNormalizingFlow(1, [50, 50, 50])
params = model.init(key, X)
z, log_jacob = model.apply(params, X)
log_p = z + log_jacob
print(jnp.any(jnp.isnan(log_p)))


#optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
#optimizer = optimizer_def.create(params)

# Compiling train step
#update = create_update(loss_fn, (model, X))
#train_max_iter(update, optimizer, None, 5000)
# print(X.shape)

# %%
