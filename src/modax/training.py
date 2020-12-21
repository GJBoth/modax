import jax
from jax import numpy as jnp
from functools import partial
from modax.losses import mse_loss


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
