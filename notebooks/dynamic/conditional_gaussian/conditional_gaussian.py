from jax import jit, numpy as jnp

from modax.training.logging import Logger
from modax.training.utils import create_stateful_update
from modax.losses.utils import normal_LL


def train(
    model,
    optimizer,
    state,
    loss_fn,
    X,
    y,
    max_epochs=1e4,
    split=0.2,
    rand_seed=42,
    **loss_fn_kwargs,
):

    # Creating update functions
    update = create_stateful_update(loss_fn, model=model, x=X, y=y, **loss_fn_kwargs)

    logger = Logger()

    for epoch in jnp.arange(max_epochs):
        (optimizer, state), metrics, output = update(optimizer, state)

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {metrics['loss']}")
        if epoch % 50 == 0:
            logger.write(metrics, epoch)

    logger.close()
    return optimizer, state


def loss_fn_conditional_preset_rho(params, state, model, x, y):
    """ Normal loss for MSE, conditional Gaussian for Reg. Correlation
    preset to 1.
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    rho = jnp.corrcoef(jnp.concatenate([prediction, dt], axis=1).T)[0, 1]
    sigma_mse = jnp.sqrt(jnp.mean((prediction - y) ** 2))
    p_mse, MSE = normal_LL(prediction, y, 1 / sigma_mse ** 2)

    dt = dt / jnp.std(dt)
    theta = theta / jnp.std(dt)
    sigma_reg = jnp.sqrt(jnp.mean((dt - theta @ coeffs) ** 2))
    mu_reg = (theta @ coeffs) + sigma_reg / sigma_mse * rho * (prediction - y)
    nu_reg = 1 / ((1 - rho ** 2) * sigma_reg ** 2)
    p_reg, reg = normal_LL(dt, mu_reg, nu_reg)

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": 1 / sigma_mse ** 2,
        "nu": sigma_reg ** 2,
        "nu_reg": nu_reg,
        "rho": rho,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_scaled(params, state, model, x, y):
    """ Normal loss for MSE, conditional Gaussian for Reg. Correlation
    preset to 1.
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    sigma_mse = jnp.sqrt(jnp.mean((prediction - y) ** 2))
    p_mse, MSE = normal_LL(prediction, y, 1 / sigma_mse ** 2)

    dt_normed = dt  # jnp.std(dt)
    theta_normed = theta  # jnp.std(dt)

    sigma_reg = jnp.sqrt(jnp.mean((dt_normed - theta_normed @ coeffs) ** 2))
    p_reg, reg = normal_LL(dt_normed, theta_normed @ coeffs, 1 / sigma_reg ** 2)

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": 1 / sigma_mse ** 2,
        "nu": 1 / sigma_reg ** 2,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_2D(params, state, model, x, y):
    """ Normal loss for MSE, conditional Gaussian for Reg. Correlation
    preset to 1.
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    rho = jnp.corrcoef(jnp.concatenate([prediction, dt], axis=1).T)[0, 1]
    sigma_mse = jnp.sqrt(jnp.mean((prediction - y) ** 2))
    sigma_reg = jnp.sqrt(jnp.mean((dt - theta @ coeffs) ** 2))

    p_mse, MSE = normal_LL(prediction, y, 1 / sigma_mse ** 2)

    dt = dt / jnp.std(dt)
    theta = theta / jnp.std(dt)

    mu_reg = (theta @ coeffs) + sigma_reg / sigma_mse * rho * (prediction - y)
    nu_reg = 1 / ((1 - rho ** 2) * sigma_reg ** 2)
    p_reg, reg = normal_LL(dt, mu_reg, nu_reg)

    loss = -(p_mse + p_reg)

    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": reg,
        "coeff": coeffs,
        "tau": 1 / sigma_mse ** 2,
        "nu": sigma_reg ** 2,
        "nu_reg": nu_reg,
        "rho": rho,
    }
    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))
