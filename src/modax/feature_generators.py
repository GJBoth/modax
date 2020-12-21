import jax
from jax import numpy as jnp
from functools import partial
from jax.ops import index_update, index


def vgrad_backward(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


def vgrad_forward(f, x, input_idx):
    s = index_update(jnp.zeros_like(x), index[:, input_idx], 1)
    _, jvp = jax.jvp(f, (x,), (s,))
    return jvp


def library_forward(f, x):
    # First derivs
    df = partial(vgrad_forward, f, input_idx=1)
    d2f = partial(vgrad_forward, df, input_idx=1)
    d3f = partial(vgrad_forward, d2f, input_idx=1)

    pred = jnp.expand_dims(f(x).T, axis=-1)
    dt = jnp.expand_dims(vgrad_forward(f, x, input_idx=0).T, axis=-1)

    # Polynomials: 1st axis is experiment, 2nd sample, 3rd dimension
    u = jnp.concatenate([jnp.ones_like(pred), pred, pred ** 2], axis=-1)
    du = jnp.concatenate(
        [
            jnp.ones_like(pred),
            jnp.expand_dims(df(x).T, axis=-1),
            jnp.expand_dims(d2f(x).T, axis=-1),
            jnp.expand_dims(d3f(x).T, axis=-1),
        ],
        axis=-1,
    )
    theta = (jnp.expand_dims(u, axis=-1) @ jnp.expand_dims(du, axis=-2)).reshape(
        *pred.shape[:2], -1
    )

    return pred.squeeze().T, dt, theta


def library_mixed(f, x):
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


def library_backward(f, x):
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

