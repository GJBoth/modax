# %% Imports
from jax import jit, numpy as jnp
from ..utils.forward_solver import fixed_point_solver


@jit
def bayesian_regression(
    X,
    y,
    prior_init=None,
    alpha_prior=(1e-6, 1e-6),
    beta_prior=(1e-6, 1e-6),
    tol=1e-5,
    max_iter=300,
):

    # Prepping matrices
    XT_y = jnp.dot(X.T, y)
    _, S, Vh = jnp.linalg.svd(X, full_matrices=False)
    eigen_vals_ = S ** 2

    n_samples, n_features = X.shape

    if prior_init is None:
        prior_init = jnp.concatenate(
            [jnp.ones((n_features + 1)), (1.0 / (jnp.var(y) + 1e-7))[jnp.newaxis]],
            axis=0,
        )
    norm_weight = jnp.concatenate((jnp.ones((n_features,)), jnp.zeros((2,))), axis=0)

    # Running
    prior_params, metrics = fixed_point_solver(
        update,
        (X, y, eigen_vals_, Vh, XT_y, alpha_prior, beta_prior),
        prior_init,
        norm_weight,
        tol=tol,
        max_iter=max_iter,
    )

    prior_params = prior_params[n_features:]  # removing coeffs
    log_LL, mn = evidence(
        X, y, prior_params, eigen_vals_, Vh, XT_y, alpha_prior, beta_prior
    )
    return log_LL, mn, prior_params, metrics


@jit
def update(
    prior,
    X,
    y,
    eigen_vals,
    Vh,
    XT_y,
    alpha_prior=(1e-6, 1e-6),
    beta_prior=(1e-6, 1e-6),
):

    n_samples, n_features = X.shape
    alpha, beta = prior[n_features:-1], prior[-1]
    # Calculating coeffs
    coeffs = jnp.linalg.multi_dot(
        [Vh.T, Vh / (eigen_vals + alpha / beta)[:, jnp.newaxis], XT_y]
    )
    gamma_ = jnp.sum((beta * eigen_vals) / (alpha + beta * eigen_vals))
    rmse_ = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)

    # %% Update alpha and lambda according to (MacKay, 1992)
    alpha = (gamma_ + 2 * alpha_prior[0]) / (jnp.sum(coeffs ** 2) + 2 * alpha_prior[1])
    beta = (n_samples - gamma_ + 2 * beta_prior[0]) / (rmse_ + 2 * beta_prior[1])
    return jnp.concatenate(
        [coeffs.squeeze(), alpha[jnp.newaxis], beta[jnp.newaxis]], axis=0
    )


@jit
def evidence(X, y, prior, eigen_vals, Vh, XT_y, alpha_prior, beta_prior):
    # compute the log of the determinant of the posterior covariance.
    # posterior covariance is given by
    # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
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

