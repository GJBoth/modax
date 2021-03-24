# %% Imports
from jax import jit, numpy as jnp
from jax._src.lax.lax import stop_gradient
from ..utils.forward_solver import fixed_point_solver
import jax


@jit
def bayesian_regression(
    X,
    y,
    prior_init=None,
    hyper_prior=((1e-6, 1e-6), (1e-6, 1e-6)),
    tol=1e-5,
    max_iter=300,
):

    n_samples, n_features = X.shape
    # Prepping matrices
    XT_y = jnp.dot(X.T, y)
    _, S, Vh = jnp.linalg.svd(X, full_matrices=False)
    eigen_vals = S ** 2

    if prior_init is None:
        prior_init = jnp.ones((2,))
        prior_init = jax.ops.index_update(prior_init, 1, 1 / (jnp.var(y) + 1e-7))

    # Running
    prior_params, iterations = fixed_point_solver(
        update,
        (X, y, eigen_vals, Vh, XT_y, hyper_prior),
        prior_init,
        lambda z_prev, z: jnp.linalg.norm(z_prev[0] - z[0]) > tol,
        max_iter=max_iter,
    )

    prior = stop_gradient(prior_params)
    log_LL, mn = evidence(X, y, prior, eigen_vals, Vh, XT_y, hyper_prior)
    metrics = (iterations, 0.0)
    return log_LL, mn, prior, metrics


@jit
def update(prior, X, y, eigen_vals, Vh, XT_y, hyper_prior):

    n_samples, n_features = X.shape
    alpha, beta = prior[:-1], prior[-1]
    alpha_prior, beta_prior = hyper_prior
    # Calculating coeffs
    coeffs = jnp.linalg.multi_dot(
        [Vh.T, Vh / (eigen_vals + alpha / beta)[:, jnp.newaxis], XT_y]
    )
    gamma_ = jnp.sum((beta * eigen_vals) / (alpha + beta * eigen_vals))
    rmse_ = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)

    # %% Update alpha and lambda according to (MacKay, 1992)
    alpha = (gamma_ + 2 * alpha_prior[0]) / (jnp.sum(coeffs ** 2) + 2 * alpha_prior[1])
    beta = (n_samples - gamma_ + 2 * beta_prior[0]) / (rmse_ + 2 * beta_prior[1])
    return jnp.stack([alpha, beta], axis=0)


@jit
def evidence(X, y, prior, eigen_vals, Vh, XT_y, hyper_prior):
    # compute the log of the determinant of the posterior covariance.
    # posterior covariance is given by
    alpha_prior, beta_prior = hyper_prior
    alpha, beta = prior[:-1], prior[-1]
    n_samples, n_features = X.shape
    coeffs = jnp.linalg.multi_dot(
        [Vh.T, Vh / (eigen_vals + alpha / beta)[:, jnp.newaxis], XT_y]
    )

    rmse = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)
    logdet_sigma = -jnp.sum(jnp.log(alpha + beta * eigen_vals))

    score = alpha_prior[0] * jnp.log(alpha) - alpha_prior[1] * alpha
    score += beta_prior[0] * jnp.log(beta) - beta_prior[1] * beta
    score += 0.5 * (
        n_features * jnp.log(alpha)
        + n_samples * jnp.log(beta)
        - beta * rmse
        - alpha * jnp.sum(coeffs ** 2)
        + logdet_sigma
        - n_samples * jnp.log(2 * jnp.pi)
    )

    return score.squeeze(), coeffs

