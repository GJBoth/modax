import jax
from jax import jit, numpy as jnp
from modax.utils.forward_solver import fixed_point_solver


def SBL(
    X,
    y,
    prior_init=None,
    alpha_prior=(1e-6, 1e-6),
    lambda_prior=(1e-6, 1e-6),
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
        (X, y, gram, XT_y, alpha_prior, lambda_prior),
        prior_init,
        norm_weight,
        tol=tol,
        max_iter=max_iter,
    )
    prior_params = prior_params[n_features:]  # removing coeffs
    return prior_params, metrics


def update_sigma(gram, alpha_, lambda_):
    sigma_inv = lambda_ * jnp.eye(gram.shape[0]) + alpha_ * gram
    L_inv = jnp.linalg.pinv(jnp.linalg.cholesky(sigma_inv))
    sigma_ = jnp.dot(L_inv.T, L_inv)
    return sigma_


def update_coeff(XT_y, alpha_, sigma_):
    coef_ = alpha_ * jnp.linalg.multi_dot([sigma_, XT_y])
    return coef_


def update(prior, X, y, gram, XT_y, alpha_prior, lambda_prior):
    n_samples, n_features = X.shape
    lambda_, alpha_ = prior[n_features:-1], prior[-1]
    sigma_ = update_sigma(gram, alpha_, lambda_)
    coef_ = update_coeff(XT_y, alpha_, sigma_)

    # Update alpha and lambda
    rmse_ = jnp.sum((y - jnp.dot(X, coef_)) ** 2)
    gamma_ = 1.0 - lambda_ * jnp.diag(sigma_)
    lambda_ = (gamma_ + 2.0 * lambda_prior[0]) / ((coef_ ** 2 + 2.0 * lambda_prior[1]))

    alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_prior[0]) / (
        rmse_ + 2.0 * alpha_prior[1]
    )

    return jnp.concatenate([coef_.squeeze(), lambda_, alpha_[jnp.newaxis]], axis=0)
