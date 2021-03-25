import jax.numpy as jnp
from .utils import normal_LL, gamma_LL, precision

# Maximum likelihood stuff
def loss_fn_mse_grad(params, state, model, x, y):
    """ first argument should always be params!
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    tau = jnp.exp(-z[0, 0])  # tau precision of mse
    nu = jnp.exp(-z[0, 1])  # precision of reg
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu)

    loss = -p_mse

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau,
        "nu": nu,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_mse_precalc(params, state, model, x, y):
    """ first argument should always be params!
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    tau_ml = 1 / jnp.mean((prediction - y) ** 2)
    p_mse, MSE = normal_LL(prediction, y, tau_ml)

    nu_ml = 1 / jnp.mean((dt - theta @ coeffs) ** 2)
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu_ml)

    loss = -p_mse

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau_ml,
        "nu": nu_ml,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_multitask_grad(params, state, model, x, y):
    """ first argument should always be params!
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    tau = jnp.exp(-z[0, 0])  # tau precision of mse
    nu = jnp.exp(-z[0, 1])  # precision of reg
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu)

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau,
        "nu": nu,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_multitask_precalc(params, state, model, x, y):
    """ first argument should always be params!
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    tau_ml = 1 / jnp.mean((prediction - y) ** 2)
    p_mse, MSE = normal_LL(prediction, y, tau_ml)

    nu_ml = 1 / jnp.mean((dt - theta @ coeffs) ** 2)
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu_ml)

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau_ml,
        "nu": nu_ml,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


# Bayesian multitask
# setting priors for these to zero should give same results as functions above.
def loss_fn_mse_bayes_grad(params, state, model, x, y, prior_params_mse=(0.0, 0.0)):
    """ first argument should always be params!
    """

    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    tau = jnp.exp(-z[0, 0])  # tau precision of mse
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_mse += gamma_LL(tau, *prior_params_mse)  # adding prior

    loss = -p_mse

    metrics = {"loss": loss, "p_mse": p_mse, "mse": MSE, "coeff": coeffs, "tau": tau}
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_mse_bayes_typeII(params, state, model, x, y, prior_params_mse=(0.0, 0.0)):
    """ first argument should always be params!
    """

    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    # Calculating precision of mse
    tau = precision(y, prediction, *prior_params_mse)
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_mse += gamma_LL(tau, *prior_params_mse)  # adding prior

    loss = -p_mse

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "coeff": coeffs,
        "tau": tau,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_pinn_bayes_grad(
    params, state, model, x, y, prior_params_reg, prior_params_mse=(0.0, 0.0)
):
    """ first argument should always be params!
    """

    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    # MSE
    tau = jnp.exp(-z[0, 0])  # tau precision of mse
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_mse += gamma_LL(tau, *prior_params_mse)  # adding prior

    # Reg
    nu = jnp.exp(-z[0, 1])  # nu precision of reg
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu)
    p_reg += gamma_LL(nu, *prior_params_reg)  # adding prior

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau,
        "nu": nu,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_pinn_bayes_typeII(
    params, state, model, x, y, prior_params_reg, prior_params_mse=(0.0, 0.0)
):

    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    # Calculating precision of mse
    tau = precision(y, prediction, *prior_params_mse)
    p_mse, MSE = normal_LL(prediction, y, tau)
    p_mse += gamma_LL(tau, *prior_params_mse)  # adding prior

    # Calculating precision of reg
    nu = precision(
        dt, theta @ coeffs, *prior_params_reg
    )  # calculates nu given gamma prior
    p_reg, reg = normal_LL(dt, theta @ coeffs, nu)
    p_reg += gamma_LL(nu, *prior_params_reg)  # adding priorr

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": tau,
        "nu": nu,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


# Fully bayesian stuff?


# Different constraints

# Bay.Reg.


# SBL
