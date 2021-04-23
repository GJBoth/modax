import jax
from jax.ops import index_update, index
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Callable


def vgrad_backward(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


def vgrad_forward(f, x, input_idx):
    s = index_update(jnp.zeros_like(x), index[:, input_idx], 1)
    _, jvp = jax.jvp(f, (x,), (s,))
    return jvp


def nth_deriv_backward(f: Callable, x: jnp.ndarray, order: int, prop_idx: int):
    """ Calculates gradient up to n-th order of f w.r.t input_idx column of x.
    prop_column is the propagated column, meaning for higher order stuff the deriv is 
    [dx^n, dx^(n-1)dy, dx^(n-1)dz, ...], Returns tensor with shapes [n_samples, n_inputs, order]"""

    assert order > 0, "Order needs to be positive integer of 1 or higher."

    def vjp(f, v):
        def _vjp(x):
            return jax.vjp(f, x)[1](v)[0]

        return _vjp

    # First order
    # Separate cause only one output
    order_vjp = vjp(f, jnp.ones((x.shape[0], 1)))
    df = [order_vjp(x)]

    # Higher order
    v = jax.ops.index_update(jnp.zeros_like(x), jax.ops.index[:, prop_idx], 1)
    for _ in np.arange(order - 1):
        order_vjp = vjp(order_vjp, v)
        df.append(order_vjp(x))
    return jnp.stack(df, axis=-1)


def nth_polynomial(x, order):
    """Returns array with [x^1, x^2, ... x^order]."""
    u = [x]
    for order in np.arange(order - 1):
        u.append(u[-1] * x)
    return jnp.concatenate(u, axis=1)
