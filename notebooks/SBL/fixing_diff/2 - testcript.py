# %% Imports
from jax import numpy as jnp, random
import jax
from modax.data.kdv import doublesoliton
from modax.models import Deepmod
from modax.training.utils import create_update
from flax import optim

from modax.training import train_max_iter
from modax.training.losses.utils import precision, normal_LL
from modax.utils.forward_solver import fixed_point_solver, fwd_solver, fwd_solver_simple


def update_sigma(gram, alpha, beta):
    sigma_inv = jnp.diag(alpha) + beta * gram
    L_inv = jnp.linalg.pinv(jnp.linalg.cholesky(sigma_inv))
    sigma_ = jnp.dot(L_inv.T, L_inv)
    return sigma_


def update_coeff(XT_y, beta, sigma_):
    coef_ = beta * jnp.linalg.multi_dot([sigma_, XT_y])
    return coef_


def update(prior, X, y, gram, XT_y, alpha_prior, beta_prior):
    n_samples, n_features = X.shape
    alpha, beta = prior[:-1], prior[-1]
    sigma = update_sigma(gram, alpha, beta)
    coeffs = update_coeff(XT_y, beta, sigma)

    # Update alpha and lambda
    rmse_ = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)
    gamma_ = 1.0 - alpha * jnp.diag(sigma)

    # TODO: Cap alpha with some threshold.
    alpha = (gamma_ + 2.0 * alpha_prior[0]) / (
        (coeffs.squeeze() ** 2 + 2.0 * alpha_prior[1])
    )
    beta = (n_samples - gamma_.sum() + 2.0 * beta_prior[0]) / (
        rmse_ + 2.0 * beta_prior[1]
    )

    return jnp.concatenate([alpha, beta[jnp.newaxis]], axis=0)


key = random.PRNGKey(42)
x = jnp.linspace(-10, 10, 100)
t = jnp.linspace(0.1, 1.0, 10)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = doublesoliton(x_grid, t_grid, c=[5.0, 2.0], x0=[0.0, -5.0])

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += 0.10 * jnp.std(y) * random.normal(key, y.shape)

# %% Building model and params
model = Deepmod([30, 30, 30, 1])
variables = model.init(key, X)

prediction, dt, theta, coeffs = model.apply(variables, X)
y = dt
X = theta

n_samples, n_features = theta.shape
prior_params_mse = (0.0, 0.0)
tau = precision(y, prediction, *prior_params_mse)

alpha_prior = (1e-6, 1e-6)
beta_prior = (n_samples / 2, n_samples / (2 * jax.lax.stop_gradient(tau)))

n_samples, n_features = X.shape
norm_weight = jnp.concatenate((jnp.ones((n_features,)), jnp.zeros((1,))), axis=0)
prior_init = jnp.concatenate(
    [jnp.ones((n_features,)), (1.0 / (jnp.var(y) + 1e-7))[jnp.newaxis]], axis=0
)
gram = jnp.dot(X.T, X)
XT_y = jnp.dot(X.T, y)

tol = 1e-3
max_iter = 1000  # low to keep it manageable


prior_params, metrics = fixed_point_solver(
    update,
    (X, y, gram, XT_y, alpha_prior, beta_prior),
    prior_init,
    norm_weight,
    tol=tol,
    max_iter=max_iter,
)

print(prior_params)
grad_fn = jax.grad(
    lambda X: fixed_point_solver(
        update,
        (X, y, gram, XT_y, alpha_prior, beta_prior),
        prior_init,
        norm_weight,
        tol=tol,
        max_iter=max_iter,
    )[0][0]
)

grads = grad_fn(X)
print(grads)
