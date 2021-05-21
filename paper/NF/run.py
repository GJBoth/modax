import jax.numpy as jnp
import numpy as np
import jax.random as random

from data import random_walk
from nf import AmortizedNormalizingFlow
from loss_models import loss_fn_SBL, NF_library

from modax.training import create_update, train_max_iter
from modax.models.models import DeepmodBase
from modax.layers.regression import LeastSquares
from flax import optim
from functools import partial


def dataset(key, n_steps, n_walkers, D, v, dt, sigma0, x0):
    def initial_condition(loc, width, key, shape):
        key1, key2 = random.split(key)
        ini_1 = loc[0] + width[0] * random.normal(key1, (shape[0], int(shape[1] / 2)))
        ini_2 = loc[1] + width[1] * random.normal(key2, (shape[0], int(shape[1] / 2)))
        return jnp.concatenate([ini_1, ini_2], axis=1)

    ini = partial(initial_condition, x0, sigma0)
    locs, t = random_walk(key, ini, n_steps, n_walkers, D, v, dt=dt)
    return jnp.expand_dims(locs, -1), jnp.expand_dims(t, -1)


key = random.PRNGKey(42)
D = 1.5
v = 0.5
dt = 0.1
sigma0 = [1.5, 0.5]
x0 = [-5, 1]
n_steps = 50
n_walkers = 100

X, t = dataset(key, n_steps, n_walkers, D, v, dt, sigma0, x0)
# X_true = np.array(dataset(key, n_steps, 10000, D, v, dt, sigma0, x0)[0])
t += 1.0  # add an offset otherwise it has lots of issues with t=0
model = DeepmodBase(
    AmortizedNormalizingFlow, ([30, 30], 10,), NF_library, (), LeastSquares, ()
)
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)

variables = model.init(key, (X, t))
state, params = variables.pop("params")
state = (state, {"prior_init": None})  # adding prior to state
optimizer = optimizer_def.create(params)
update_fn = create_update(loss_fn_SBL, (model, (X, t)))
optimizer, state = train_max_iter(update_fn, optimizer, state, 20000)

jnp.save("trained_model", optimizer.target)

# Sampling on a grid for nice pictures
x = jnp.linspace(-10, 10, 200)
x_grid = jnp.meshgrid(x, t.squeeze())[0][:, :, None]

model_state, loss_state = state
log_p, dt, theta, coeffs = model.apply(
    {"params": optimizer.target, **model_state}, (x_grid, t)
)
p = jnp.exp(log_p).reshape(x_grid.shape[:2])

jnp.save("inferred_density.npy", p)

