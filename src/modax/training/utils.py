from jax import jit, value_and_grad, numpy as jnp
import numpy as np


def create_update(loss_fn, *args, **kwargs):
    """Constructs a fast update given a loss function."""

    def step(opt, loss_fn, *args, **kwargs):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, metrics), grad = grad_fn(opt.target, *args, **kwargs)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, metrics

    return jit(lambda opt: step(opt, loss_fn, *args, **kwargs))


def create_stateful_update(loss_fn, *args, **kwargs):
    def step(opt, state, loss_fn, *args, **kwargs):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (updated_state, metrics, output)), grad = grad_fn(
            opt.target, state, *args, **kwargs
        )
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.

        return (opt, updated_state), metrics, output

    return jit(lambda opt, state: step(opt, state, loss_fn, *args, **kwargs))

