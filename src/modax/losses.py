import jax
from jax import numpy as jnp


def mse_loss(y_pred, y):
    def squared_error(y, y_pred):
        return jnp.inner(y - y_pred, y - y_pred) / 2.0

    return jnp.mean(jax.vmap(squared_error)(y, y_pred), axis=0)
