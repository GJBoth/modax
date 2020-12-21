import jax
from typing import Sequence, Callable
from jax import numpy as jnp, lax
from flax import linen as nn
from jax.scipy.special import erfc
from functools import partial
from jax.ops import index_update, index
from modax.layers import MultiTaskDense


def mse_loss(y_pred, y):
    def squared_error(y, y_pred):
        return jnp.inner(y - y_pred, y - y_pred) / 2.0

    return jnp.mean(jax.vmap(squared_error)(y, y_pred), axis=0)


def burgers(x, t, v, A):
    R = A / (2 * v)
    z = x / jnp.sqrt(4 * v * t)

    u = (
        jnp.sqrt(v / (jnp.pi * t))
        * ((jnp.exp(R) - 1) * jnp.exp(-(z ** 2)))
        / (1 + (jnp.exp(R) - 1) / 2 * erfc(z))
    )
    return u


def library_jvp(f, x):
    # First derivs
    df = partial(vgrad_backward, f)
    d2f = partial(vgrad_forward, lambda y: df(y)[:, [1]], input_idx=1)
    d3f = partial(vgrad_forward, d2f, input_idx=1)

    pred = f(x)
    dt, dx = df(x).T
    u = jnp.concatenate([jnp.ones_like(pred), pred, pred ** 2], axis=1)
    du = jnp.concatenate([jnp.ones_like(pred), dx[:, None], d2f(x), d3f(x)], axis=1)
    theta = (u[:, :, None] @ du[:, None, :]).reshape(
        -1, 12
    )  # maybe rewrite using vmap?
    return pred, dt[:, None], theta


def vgrad_backward(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


def vgrad_forward(f, x, input_idx):
    s = index_update(jnp.zeros_like(x), index[:, input_idx], 1)
    _, jvp = jax.jvp(f, (x,), (s,))
    return jvp


def library_vjp(f, x):
    """library using only jvp"""
    # First derivs
    df = partial(vgrad_backward, f)
    d2f = partial(vgrad_backward, lambda y: df(y)[:, [1]])
    d3f = partial(vgrad_backward(), lambda y: d2f(y)[:, [1]])

    pred = f(x)
    dt, dx = df(x).T
    u = jnp.concatenate([jnp.ones_like(pred), pred, pred ** 2], axis=1)
    du = jnp.concatenate(
        [jnp.ones_like(pred), dx[:, None], d2f(x)[:, [1]], d3f(x)[:, [1]]], axis=1
    )
    theta = (u[:, :, None] @ du[:, None, :]).reshape(
        -1, 12
    )  # maybe rewrite using vmap?
    return pred, dt[:, None], theta

