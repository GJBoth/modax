from jax import numpy as jnp
from functools import partial
from .utils import nth_deriv_backward, vgrad_backward, vgrad_forward, nth_polynomial


def library_forward(f, x):
    # First derivs
    df = partial(vgrad_forward, f, input_idx=1)
    d2f = partial(vgrad_forward, df, input_idx=1)
    d3f = partial(vgrad_forward, d2f, input_idx=1)

    pred = jnp.expand_dims(f(x).T, axis=-1)
    dt = jnp.expand_dims(
        vgrad_forward(f, x, input_idx=0).T, axis=-1
    )  # TODO: Together with x as matrix jacobian

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
    )  # TODO: make this into scan?

    theta = (jnp.expand_dims(u, axis=-1) @ jnp.expand_dims(du, axis=-2)).reshape(
        u.shape[0], u.shape[1], 12
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
    d3f = partial(vgrad_backward, lambda y: d2f(y)[:, [1]])

    pred = f(x)
    dt, dx = df(x).T
    u = jnp.concatenate([jnp.ones_like(pred), pred, pred ** 2], axis=1)
    du = jnp.concatenate(
        [jnp.ones_like(pred), dx[:, None], d2f(x)[:, [1]], d3f(x)[:, [1]]], axis=1
    )
    theta = (u[:, :, None] @ du[:, None, :]).reshape(
        -1, 12
    )  # maybe rewrite using vmap?
    return pred, (dt[:, None], theta)


def library_backward_new(deriv_order, poly_order):
    def library_fn(f, x):
        # polynomial part
        pred = f(x)
        polynomials = poly_fn(pred)

        # derivative part
        df = deriv_fn(f, x)
        dt = df[:, 0, [0]]  # time
        derivs = df[:, 1, :]  # space

        # Making library
        u = jnp.concatenate([jnp.ones_like(pred), polynomials], axis=1)[:, :, None]
        du = jnp.concatenate([jnp.ones_like(pred), derivs], axis=1)[:, None, :]
        n_features = (deriv_order + 1) * (poly_order + 1)
        theta = jnp.matmul(u, du).reshape(-1, n_features)
        return pred, (dt, theta)

    deriv_fn = partial(nth_deriv_backward, order=deriv_order, prop_idx=1)
    poly_fn = partial(nth_polynomial, order=poly_order)
    return library_fn
