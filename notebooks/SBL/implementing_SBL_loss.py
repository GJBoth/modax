# %% Imports
from jax.api import value_and_grad
from jax.config import config

config.update("jax_debug_nans", True)

from jax import numpy as jnp, random
import jax
from modax.data.burgers import burgers
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training.losses.SBL import loss_fn_SBL
from modax.training import train_max_iter
from sklearn.linear_model import ARDRegression
from modax.linear_model.SBL import SBL

# %% Making data
key = random.PRNGKey(42)

x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.5, 5.0, 20)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += 0.10 * jnp.std(y) * random.normal(key, y.shape)

# %% Building model and params
model = Deepmod([30, 30, 30, 1])
variables = model.init(key, X)

optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)

state = (state, {"prior_init": None})  # adding prior to state
update_fn = create_update(loss_fn_SBL, (model, X, y, False))

# optimizer, state = train_max_iter(update_fn, optimizer, state, 10000)

grad_fn = jax.value_and_grad(loss_fn_SBL, has_aux=True)
(loss, (updated_state, metrics, output)), grad = grad_fn(
    optimizer.target, state, model, X, y
)


# %%

# %%
model_state, loss_state = state
variables = {"params": params, **model_state}
(prediction, dt, theta, coeffs), updated_model_state = model.apply(
    variables, X, mutable=list(model_state.keys())
)

theta_normed = theta / jnp.linalg.norm(theta, axis=0)
# %%
n_samples, n_features = theta.shape
tau = 1 / jnp.mean((y - prediction) ** 2)
hyper_prior = (n_samples / 2, n_samples / 2 * 1 / tau)
# hyper_prior = (1e-6, 1e-6)
# %%
reg = ARDRegression(
    fit_intercept=False,
    compute_score=True,
    tol=1e-3 * tau,
    alpha_1=hyper_prior[0],
    alpha_2=hyper_prior[1],
    threshold_lambda=1e6,
)
reg.fit(theta_normed, dt)

# %%
loss, mn, prior, metric = SBL(theta_normed, dt, beta_prior=hyper_prior, tol=1e-3 * tau)
# %%
# hyper_prior = (n_samples / 2, n_samples / 2 * 1 / (1e-6))

hyper_prior = (1e-6, 1e-6)
f = lambda x, y: SBL(x, y, beta_prior=hyper_prior, tol=1e-3)[0]
grad_fn = value_and_grad(f)
grad_fn(theta_normed, dt)
# %%

# %%
hyper_prior = (1e-6, 1e-6)
f = lambda x, y: SBL(x, y, beta_prior=hyper_prior, tol=1e-3)[0].sum()
grad_fn = value_and_grad(f)
grad_fn(theta_normed, dt)
# %%

n_samples, n_features = theta.shape
prior_init = jnp.concatenate(
    [jnp.ones((n_features,)), (1.0 / (jnp.var(y) + 1e-7))[jnp.newaxis]], axis=0
)
# %%
# adding zeros to z for coeffs
# prior_init = jnp.concatenate([jnp.zeros((n_features,)), prior_init], axis=0)
norm_weight = jnp.concatenate((jnp.ones((n_features,)), jnp.zeros((1,))), axis=0)
gram = jnp.dot(X.T, X)
XT_y = jnp.dot(X.T, y)

prior_params, metrics = fixed_point_solver(
    update,
    (X, y, gram, XT_y, alpha_prior, beta_prior, cap),
    prior_init,
    norm_weight,
    tol=tol,
    max_iter=max_iter,
)
