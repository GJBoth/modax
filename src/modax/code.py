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


def train_step(model, x, y):
    """Constructs a fast update given a loss function.
    """

    def step(opt, x, y, model):
        def loss_fn(params):
            prediction = model.apply(params, x)
            loss = mse_loss(prediction, y)

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(opt.target)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(lambda opt: step(opt, x, y, model))


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


def train_step_pinn(model, library, x, y):
    """Constructs a fast update given a loss function.
    """

    def step(opt, x, y, model, library):
        def loss_fn(params, x, y, model, library):
            prediction, dt, theta = library(partial(model.apply, params), x)

            fit = lambda X, y: jnp.linalg.lstsq(X, y)[0]
            coeffs = jax.vmap(fit, in_axes=(0, 0), out_axes=0)(
                theta, dt
            )  # vmap over experiments

            loss = mse_loss(prediction, y) + mse_loss(
                dt.squeeze(), (theta @ coeffs).squeeze()
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, x, y, model, library)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(partial(step, x=x, y=y, model=model, library=library))


def train_step_pinn_mt(model, library, x, y):
    """Constructs a fast update given a loss function.
    """

    def step(opt, x, y, model, library):
        def loss_fn(params, x, y, model, library):
            f = lambda x: model.apply(params, x).squeeze().T
            prediction, dt, theta = library(f, x)

            fit = lambda X, y: jnp.linalg.lstsq(X, y)[0]
            coeffs = jax.vmap(fit, in_axes=(0, 0), out_axes=0)(
                theta, dt
            )  # vmap over experiments

            loss = mse_loss(prediction, y) + mse_loss(
                dt.squeeze(), (theta @ coeffs).squeeze()
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, x, y, model, library)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(partial(step, x=x, y=y, model=model, library=library))
