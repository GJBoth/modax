import jax
from jax import jit, numpy as jnp
from modax.utils.forward_solver import fixed_point_solver
from jax.scipy.linalg import solve_triangular
from jax.numpy.linalg import cholesky
from jax.lax import stop_gradient


def update_posterior(gram, XT_y, prior):
    alpha, beta = prior
    L = cholesky(jnp.diag(alpha) + beta * gram)
    R = solve_triangular(L, jnp.eye(alpha.shape[0]), check_finite=False, lower=True)
    sigma = jnp.dot(R.T, R)
    mean = beta * jnp.dot(sigma, XT_y)

    return mean, sigma


def update_prior(X, y, posterior, prior, hyper_prior):
    n_samples, n_features = X.shape
    mean, covariance = posterior
    alpha, beta = prior
    alpha_prior, beta_prior = hyper_prior

    rmse = jnp.sum((y - jnp.dot(X, mean)) ** 2)
    gamma = 1.0 - alpha * jnp.diag(covariance)

    # Update alpha and beta
    alpha = (gamma + 2.0 * alpha_prior[0]) / (
        (mean.squeeze() ** 2 + 2.0 * alpha_prior[1])
    )
    beta = (n_samples - jnp.sum(gamma) + 2.0 * beta_prior[0]) / (
        rmse + 2.0 * beta_prior[1]
    )

    # Calculating dL/da
    dLda = (
        1
        / alpha
        * (
            1 / 2 * (1 - alpha * (mean.squeeze() ** 2 + jnp.diag(covariance)))
            + alpha_prior[0]
            - alpha * alpha_prior[1]
        )
    )
    return jnp.minimum(1e7, alpha), jnp.minimum(1e7, beta), jnp.mean(jnp.abs(dLda))


def update(prior, X, y, gram, XT_y, hyper_prior):
    alpha, beta = prior[:-2], prior[-2]
    posterior = update_posterior(gram, XT_y, (alpha, beta))
    alpha, beta, loss_grad = update_prior(X, y, posterior, (alpha, beta), hyper_prior)

    return jnp.concatenate([alpha, beta[jnp.newaxis], loss_grad[jnp.newaxis]], axis=0)


def evidence(X, y, gram, XT_y, prior, hyper_prior):
    n_samples, n_features = X.shape
    alpha, beta = prior[:-1], prior[-1]
    alpha_prior, beta_prior = hyper_prior

    mean, covariance = update_posterior(gram, XT_y, (alpha, beta))
    rmse = jnp.sum((y - jnp.dot(X, mean)) ** 2)

    score = jnp.sum(alpha_prior[0] * jnp.log(alpha) - alpha_prior[1] * alpha)
    score += beta_prior[0] * jnp.log(beta) - beta_prior[1] * beta
    score += 0.5 * (
        jnp.linalg.slogdet(covariance)[1]
        + n_samples * jnp.log(beta)
        + jnp.sum(jnp.log(alpha))
    )
    score -= 0.5 * (beta * rmse + jnp.sum(alpha * mean.squeeze() ** 2))

    return score.squeeze(), mean


@jit
def SBL(
    X,
    y,
    prior_init=None,
    hyper_prior=((1e-6, 1e-6), (1e-6, 1e-6)),
    tol=1e-5,
    max_iter=1000,
    stop_prior_grad=True,
):

    n_samples, n_features = X.shape
    if prior_init is None:
        prior_init = jnp.ones((n_features + 2,))
        prior_init = jax.ops.index_update(
            prior_init, n_features, 1 / (jnp.var(y) + 1e-6)
        )  # setting initial noise value

    gram = jnp.dot(X.T, X)
    XT_y = jnp.dot(X.T, y)

    prior_params, metrics = fixed_point_solver(
        update,
        (X, y, gram, XT_y, hyper_prior),
        prior_init,
        lambda z_prev, z: z[-1] > tol,
        max_iter=max_iter,
    )

    prior = jax.lax.cond(
        stop_prior_grad, lambda x: stop_gradient(x[:-1]), lambda x: x[:-1], prior_params
    )

    loss, mn = evidence(X, y, gram, XT_y, prior, hyper_prior)
    return loss, mn, prior, metrics
