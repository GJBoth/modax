import jax
from jax import jit, numpy as jnp
from modax.utils.forward_solver import fixed_point_solver


def SBL(
    X,
    y,
    prior_init=None,
    alpha_prior=(1e-6, 1e-6),
    beta_prior=(1e-6, 1e-6),
    tol=1e-3,
    max_iter=300,
):
    n_samples, n_features = X.shape
    norm_weight = jnp.concatenate(
        (jnp.ones((n_features,)), jnp.zeros((n_features + 1,))), axis=0
    )
    if prior_init is None:
        prior_init = jnp.concatenate(
            [jnp.ones((n_features,)), (1.0 / (jnp.var(y) + 1e-7))[jnp.newaxis]], axis=0
        )
    # adding zeros to z for coeffs
    prior_init = jnp.concatenate([jnp.zeros((n_features,)), prior_init], axis=0)
    gram = jnp.dot(X.T, X)
    XT_y = jnp.dot(X.T, y)

    prior_params, metrics = fixed_point_solver(
        update,
        (X, y, gram, XT_y, alpha_prior, beta_prior),
        prior_init,
        norm_weight,
        tol=tol,
        max_iter=max_iter,
    )
    prior_params = prior_params[n_features:]  # removing coeffs
    return prior_params, metrics


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
    alpha, beta = prior[n_features:-1], prior[-1]
    sigma = update_sigma(gram, alpha, beta)
    coeffs = update_coeff(XT_y, beta, sigma)

    # Update alpha and lambda
    rmse_ = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)
    gamma_ = 1.0 - alpha * jnp.diag(sigma)
    alpha = (gamma_ + 2.0 * alpha_prior[0]) / ((coeffs ** 2 + 2.0 * alpha_prior[1]))

    beta = (n_samples - gamma_.sum() + 2.0 * beta_prior[0]) / (
        rmse_ + 2.0 * beta_prior[1]
    )

    return jnp.concatenate([coeffs.squeeze(), alpha, beta[jnp.newaxis]], axis=0)
