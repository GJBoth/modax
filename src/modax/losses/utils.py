import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def mse(y_pred, y):
    """Helper fuction to calculate MSE.
    """

    def squared_error(y, y_pred):
        return jnp.inner(y - y_pred, y - y_pred) / 2.0

    return jnp.mean(jax.vmap(squared_error)(y, y_pred), axis=0)


def neg_LL(pred, loc, precision):
    sigma = jnp.sqrt(1 / precision)
    return -jnp.sum(norm.logpdf(pred, loc=loc, scale=sigma))

