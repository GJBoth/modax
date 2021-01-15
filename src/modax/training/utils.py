from functools import partial
from jax import jit, value_and_grad
from ..losses.utils import normal_LL, gamma_LL


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


def validation_metric(
    params, state, model, x, y, prior_params_mse, prior_params_reg, train_params
):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )
    tau, nu = train_params
    # Calculating precision of mse
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_mse += gamma_LL(tau, *prior_params_mse)  # adding prior

    # Calculating precision of reg
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu)
    p_reg += gamma_LL(nu, *prior_params_reg)  # adding priorr

    metrics = {
        "val_p_mse": p_mse,
        "val_mse": MSE,
        "val_p_reg": p_reg,
        "val_reg": reg,
    }
    return metrics
