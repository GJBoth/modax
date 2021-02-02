# %% Imports
from jax import jit, numpy as jnp
from modax.utils.forward_solver import fixed_point_solver


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
    def update(
        prior_params,
        X,
        y,
        eigen_vals_,
        Vh,
        XT_y,
        alpha_prior=(1e-6, 1e-6),
        beta_prior=(1e-6, 1e-6),
    ):
        alpha, beta = prior_params[:-1], prior_params[-1]
        n_samples = X.shape[0]

        # Calculating coeffs
        coeffs = jnp.linalg.multi_dot(
            [Vh.T, Vh / (eigen_vals_ + alpha / beta)[:, jnp.newaxis], XT_y]
        )
        gamma_ = jnp.sum((beta * eigen_vals_) / (alpha + beta * eigen_vals_))
        rmse_ = jnp.sum((y - jnp.dot(X, coeffs)) ** 2)

        # %% Update alpha and lambda according to (MacKay, 1992)
        alpha = (gamma_ + 2 * alpha_prior[0]) / (
            jnp.sum(coeffs ** 2) + 2 * alpha_prior[1]
        )
        beta = (n_samples - gamma_ + 2 * beta_prior[0]) / (rmse_ + 2 * beta_prior[1])
        return jnp.stack([alpha, beta])

    # Prepping matrices
    XT_y = jnp.dot(X.T, y)
    _, S, Vh = jnp.linalg.svd(X, full_matrices=False)
    eigen_vals_ = S ** 2

    if prior_init is None:
        alpha = 1.0
        beta = 1.0 / (jnp.var(y) + 1e-7)
        prior_init = jnp.stack([alpha, beta])

    # Running
    prior_params, metrics = fixed_point_solver(
        update,
        (X, y, eigen_vals_, Vh, XT_y, alpha_prior, beta_prior),
        prior_init,
        tol=tol,
        max_iter=max_iter,
    )
    return prior_params, metrics


@jit
def evidence(X, y, prior_params, hyper_prior_params=(0.0, 0.0)):
    alpha, beta = prior_params[:-1], prior_params[-1]
    a, b = hyper_prior_params

    n_samples, n_terms = X.shape
    A = alpha * jnp.eye(n_terms) + beta * X.T @ X
    mn = beta * jnp.linalg.inv(A) @ X.T @ y  # get rid of inverse?

    E = beta * jnp.sum((y - X @ mn) ** 2) + alpha * jnp.sum(mn ** 2)
    loss = 0.5 * (
        n_terms * jnp.log(alpha)
        + n_samples * jnp.log(beta)
        - E
        - jnp.linalg.slogdet(A)[1]
        - n_samples * jnp.log(2 * jnp.pi)
    )

    # following tipping, numerically stable if a, b -> 0 but doesn't have constant terms.
    loss += a * jnp.log(beta) - b * beta
    return loss.squeeze(), mn

