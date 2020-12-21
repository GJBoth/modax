from jax import numpy as jnp
from jax.scipy.special import erfc


def burgers(x, t, v, A):
    R = A / (2 * v)
    z = x / jnp.sqrt(4 * v * t)

    u = (
        jnp.sqrt(v / (jnp.pi * t))
        * ((jnp.exp(R) - 1) * jnp.exp(-(z ** 2)))
        / (1 + (jnp.exp(R) - 1) / 2 * erfc(z))
    )
    return u
