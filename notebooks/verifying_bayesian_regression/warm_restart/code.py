from modax.training.losses.utils import precision, normal_LL
from modax.linear_model.bayesian_regression import bayesianregression, evidence
import jax.numpy as jnp
import jax


def loss_fn_bayesian_ridge(params, state, model, X, y, warm_restart=True):
    model_state, loss_state = state
    variables = {"params": params, **model_state}
    (prediction, dt, theta, coeffs), updated_model_state = model.apply(
        variables, X, mutable=list(model_state.keys())
    )

    n_samples = theta.shape[0]
    prior_params_mse = (0.0, 0.0)

    # MSE stuff
    tau = precision(y, prediction, *prior_params_mse)
    p_mse, MSE = normal_LL(prediction, y, tau)

    # Regression stuff
    # we dont want the gradient
    hyper_prior_params = (
        n_samples / 2,
        n_samples / (2 * jax.lax.stop_gradient(tau)),
    )
    theta_normed = theta / jnp.linalg.norm(theta, axis=0)

    if (loss_state["prior_init"] is None) or (warm_restart is False):
        prior_init = jnp.stack([1.0, 1.0 / jnp.var(dt)])
    else:
        prior_init = loss_state["prior_init"]

    prior, fwd_metric = bayesianregression(
        theta_normed,
        dt,
        prior_params_init=prior_init,
        hyper_prior_params=hyper_prior_params,
    )
    p_reg, mn = evidence(theta_normed, dt, prior, hyper_prior_params=hyper_prior_params)
    Reg = jnp.mean((dt - theta_normed @ mn) ** 2)

    loss_state["prior_init"] = prior
    loss = -(p_mse + p_reg)
    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": MSE,
        "p_reg": p_reg,
        "reg": Reg,
        "bayes_coeffs": mn,
        "coeffs": coeffs,
        "alpha": prior[:-1],
        "beta": prior[-1],
        "tau": tau,
        "its": fwd_metric[0],
        "gap": fwd_metric[1],
    }

    return (
        loss,
        ((updated_model_state, loss_state), metrics, (prediction, dt, theta, mn)),
    )
