import jax
from jax import numpy as jnp
from functools import partial
from modax.losses import mse


def create_update(loss_fn, *args, **kwargs):
    """Constructs a fast update given a loss function.
    """

    def step(opt, loss_fn, *args, **kwargs):
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, *args, **kwargs)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(lambda opt: step(opt, loss_fn, *args, **kwargs))


def train_step_pinn(model, library, constraint, x, y):
    """Constructs a fast update given a loss function.
    """

    def step(opt, x, y, model, library, constraint):
        def loss_fn(params, x, y, model, library, constraint):
            prediction, dt, theta = library(partial(model.apply, params), x)
            reg_loss, coeffs = constraint((theta, dt))
            loss = mse(prediction, y) + reg_loss
            return loss

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, x, y, model, library, constraint)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(
        partial(step, x=x, y=y, model=model, library=library, constraint=constraint)
    )


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

            loss = mse(prediction, y) + mse(dt.squeeze(), (theta @ coeffs).squeeze())
            return loss

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, x, y, model, library)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jax.jit(partial(step, x=x, y=y, model=model, library=library))
