# %% Imports
from jax import jit, numpy as jnp
from ..utils.forward_solver import fixed_point_solver


@jit
def bayesianregression(
    X,
    y,
    prior_params_init=None,
    hyper_prior_params=jnp.zeros((2,)),
    tol=1e-4,
    max_iter=300,
):
    def update(prior_params, X, y, eigvals, hyper_prior_params):
        # Unpacking parameters
        alpha_prev, beta_prev = prior_params[:-1], prior_params[-1]
        a, b = hyper_prior_params

        # Calculating intermediate matrices
        n_samples, n_terms = X.shape
        gamma_ = jnp.sum((beta_prev * eigvals) / (alpha_prev + beta_prev * eigvals))
        S = jnp.linalg.inv(
            beta_prev * X.T @ X + alpha_prev * jnp.eye(n_terms)
        )  # remove inverse?
        mn = beta_prev * S @ X.T @ y

        # Update estimate
        alpha = gamma_ / jnp.sum(mn ** 2)
        beta = (n_samples - gamma_ + 2 * a) / (jnp.sum((y - X @ mn) ** 2) + 2 * b)

        return jnp.stack([alpha, beta])

    # Constructing update function.
    eigvals = jnp.linalg.eigvalsh(X.T @ X)

    if prior_params_init is None:
        prior_params_init = jnp.stack([1.0, 1.0 / jnp.var(y)])

    # Calculating optimal prior
    prior_params, metrics = fixed_point_solver(
        update,
        (X, y, eigvals, hyper_prior_params),
        prior_params_init,
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

