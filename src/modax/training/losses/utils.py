import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def mse(y_pred, y):
    def squared_error(y, y_pred):
        return jnp.inner(y - y_pred, y - y_pred)

    return jnp.mean(jax.vmap(squared_error)(y, y_pred), axis=0)


def normal_LL(x, mu, tau):
    # tau = 1 / sigma**2, for numerical reasons.
    n_samples = x.shape[0]

    mse = 1 / 2 * jnp.mean((x - mu) ** 2)
    # 2 before tau to compensate for 1/2
    p = -n_samples / 2 * (2 * tau * mse - jnp.log(tau) + jnp.log(2 * jnp.pi))
    return p, mse


def gamma_LL(x, alpha, beta):
    # log pdf of gamma dist, dropped constant factors since it can be improper.
    p = (alpha - 1) * jnp.log(x) - beta * x
    return p


def precision(y, x, alpha, beta):
    # calculates precision parameter with a gamma prior
    n_samples = y.shape[0]
    tau = (n_samples + 2 * (alpha - 1)) / (jnp.linalg.norm(y - x) ** 2 + 2 * beta)
    return tau
