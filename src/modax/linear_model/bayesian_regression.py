# %% Imports
from jax import jit, numpy as jnp
from ..utils.forward_solver import fixed_point_solver


@jit
def bayesianregression(
    X, y, prior_params_init=None, hyper_prior_params=(0.0, 0.0), tol=1e-4, max_iter=300
):
    def update(prior_params, X, y, eigvals, norms, hyper_prior_params):
        # Unpacking parameters
        alpha_prev, beta_prev = prior_params
        a, b = hyper_prior_params

        # Calculating intermediate matrices
        n_samples, _ = X.shape
        gamma_ = jnp.sum(
            (beta_prev * eigvals) / (alpha_prev * norms + beta_prev * eigvals)
        )
        S = jnp.linalg.inv(beta_prev * X.T @ X + jnp.diag(alpha_prev * norms))
        mn = beta_prev * S @ X.T @ y

        # Update estimate
        alpha = gamma_ / jnp.sum(mn ** 2)
        beta = (n_samples - gamma_ + 2 * (a - 1)) / (jnp.sum((y - X @ mn) ** 2) + 2 * b)

        return (alpha, beta)

    # Constructing update function.
    eigvals = jnp.linalg.eigvalsh(X.T @ X)
    norms = jnp.linalg.norm(X, axis=0)

    if prior_params_init is None:
        prior_params_init = (1.0, 1.0 / jnp.var(y))

    # Calculating optimal prior
    prior_params = fixed_point_solver(
        update, (X, y, eigvals, norms, hyper_prior_params), prior_params_init, tol=tol,
    )

    return prior_params


@jit
def evidence(X, y, prior_params, hyper_prior_params):
    alpha, beta = prior_params
    a, b = hyper_prior_params

    alpha *= jnp.linalg.norm(X, axis=1)  # add the norms

    n_samples, _ = X.shape
    A = jnp.diag(alpha) + beta * X.T @ X
    mn = beta * jnp.linalg.inv(A) @ X.T @ y

    E = beta * jnp.sum((y - X @ mn) ** 2) + jnp.sum(alpha * mn ** 2)
    loss = 0.5 * (
        jnp.sum(jnp.log(alpha))
        + n_samples * jnp.log(beta)
        - E
        - jnp.linalg.slogdet(A)[1]
        - n_samples * jnp.log(2 * jnp.pi)
    )

    # following tipping, numerically stable if a, b -> 0 but doesn't have constant terms.
    loss += (a - 1) * jnp.log(beta) - b * beta
    return loss, mn

