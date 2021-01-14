import jax.numpy as jnp
from modax.losses.utils import neg_LL
from jax.scipy.stats import gamma, t


def loss_fn_mse_bayes(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs, s, t = model.apply(params, x)
    tau = jnp.exp(-s)  # tau precision of mse

    MSE = neg_LL(prediction, y, tau)
    loss = MSE

    metrics = {"loss": loss, "mse": MSE, "coeff": coeffs, "tau": tau}
    return loss, metrics


def loss_fn_multitask(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs, s, t = model.apply(params, x)
    tau = jnp.exp(-s)
    beta = jnp.exp(-t)

    MSE = neg_LL(prediction, y, tau)
    Reg = neg_LL(dt.squeeze(), (theta @ coeffs).squeeze(), beta)
    loss = MSE + Reg

    metrics = {
        "loss": loss,
        "mse": MSE,
        "reg": Reg,
        "coeff": coeffs,
        "beta": beta,
        "tau": tau,
    }
    return loss, metrics


def loss_fn_pinn_bayes(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs, s, t = model.apply(params, x)
    tau = jnp.exp(-s)
    beta = jnp.exp(-t)

    MSE = neg_LL(prediction, y, tau)
    Reg = neg_LL(dt.squeeze(), (theta @ coeffs).squeeze(), beta)
    n_samples = prediction.shape[0]
    prior = -jnp.sum(
        gamma.logpdf(beta, a=n_samples / 2, scale=1 / (n_samples / 2 * 1e-4))
    )
    loss = MSE + Reg + prior

    metrics = {
        "loss": loss,
        "mse": MSE,
        "reg": Reg,
        "coeff": coeffs,
        "beta": beta,
        "tau": tau,
    }
    return loss, metrics


def loss_fn_mse_bayes_precalc(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs, s, t = model.apply(params, x)

    sigma_ml = jnp.mean((prediction - y) ** 2)
    tau = 1 / sigma_ml
    MSE = neg_LL(prediction, y, tau)
    loss = MSE

    metrics = {"loss": loss, "mse": MSE, "coeff": coeffs, "tau": tau}
    return loss, metrics


def loss_fn_pinn_bayes_full(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs = model.apply(params, x)[:-2]
    n_samples = prediction.shape[0]
    a0, b0 = n_samples / 2, 1 / (n_samples / 2 * 1e-4)
    a_post, b_post = (
        a0 + n_samples / 2,
        jnp.sqrt(b0 + 1 / 2 * jnp.sum(dt - theta @ coeffs)),
    )

    MSE = -jnp.sum(
        t.logpdf(
            prediction,
            2 * n_samples / 2,
            loc=y,
            scale=jnp.sqrt(jnp.mean((prediction - y) ** 2)),
        )
    )

    Reg = -jnp.sum(t.logpdf(dt, 2 * a_post, loc=theta @ coeffs, scale=b_post / a_post))
    loss = MSE + Reg

    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs}
    return loss, metrics


def loss_fn_pinn_bayes_approximate(params, model, x, y):
    """ first argument should always be params!
    """
    prediction, dt, theta, coeffs = model.apply(params, x)[:-2]
    n_samples = prediction.shape[0]
    a0, b0 = n_samples / 2, 1 / (n_samples / 2 * 1e-4)

    sigma_ml = jnp.mean((prediction - y) ** 2)
    tau = 1 / sigma_ml
    MSE = neg_LL(prediction, y, tau)

    sigma_ml = jnp.mean((dt - theta @ coeffs) ** 2)
    beta = (n_samples + 2 * (a0 - 1)) / (n_samples * sigma_ml + 2 * b0)
    Reg = neg_LL(dt, theta @ coeffs, beta)

    prior = -jnp.sum(gamma.logpdf(beta, a=a0, scale=b0))

    loss = MSE + Reg + prior

    metrics = {
        "loss": loss,
        "mse": MSE,
        "reg": Reg,
        "coeff": coeffs,
        "tau": tau,
        "beta": beta,
    }
    return loss, metrics
