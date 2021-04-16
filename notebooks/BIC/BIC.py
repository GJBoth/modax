from modax.training.losses.utils import precision, normal_LL
from modax.linear_model.SBL import SBL
import jax.numpy as jnp
import jax
from jax import jit


def BIC(prediction, y, u_t, theta, alpha, alpha_threshold=1e4):
    n_samples = theta.shape[0]

    # MSE part
    mse = jnp.mean((prediction - y) ** 2)
    L1 = n_samples * jnp.log(mse)

    # Reg part
    thresholds = jnp.minimum(
        jnp.sort(alpha[:-1]) + 0.5 * jnp.diff(jnp.sort(alpha)), 1e5
    )
    masks = jnp.stack([alpha < threshold for threshold in thresholds])
    regs = (
        jax.vmap(
            lambda mask: jnp.linalg.lstsq(theta * jnp.where(mask == 0, 1e-7, 1.0), u_t)[
                1
            ]
        )(masks).squeeze()
        / n_samples
    )

    L2 = n_samples * jnp.log(regs)
    BIC = (L1 + L2) + jnp.sum(masks, axis=1) * jnp.log(n_samples)

    # Do with standard threshold
    mask = alpha < alpha_threshold
    coeffs, reg = jnp.linalg.lstsq(theta * jnp.where(mask == 0, 1e-7, 1.0), u_t)[:2]
    reg = reg / n_samples
    coeffs = coeffs.squeeze() * mask

    return (
        BIC.squeeze(),
        (mse.squeeze(), reg.squeeze()),
        coeffs,
    )


def loss_fn_SBL(params, state, model, X, y, warm_restart=True):
    model_state, loss_state = state
    variables = {"params": params, **model_state}
    (prediction, dt, theta, coeffs), updated_model_state = model.apply(
        variables, X, mutable=list(model_state.keys())
    )

    n_samples, n_features = theta.shape

    # MSE stuff
    tau = precision(y, prediction, 0.0, 0.0)
    p_mse, MSE = normal_LL(prediction, y, tau)

    # Regression stuff
    # we dont want the gradient
    beta_prior = (
        n_samples / 2,
        n_samples / (jax.lax.stop_gradient(tau)),
    )

    if warm_restart:
        prior_init = loss_state["prior_init"]
    else:
        prior_init = None

    p_reg, mn, prior, fwd_metric = SBL(
        theta,
        dt,
        prior_init=prior_init,
        hyper_prior=((1e-6, 1e-6), beta_prior),
        tol=1e-4,
        max_iter=2000,
    )

    BIC_val, (mse, reg), masked_coeffs = BIC(prediction, y, dt, theta, prior[:-1], 1e4)
    updated_loss_state = {"prior_init": prior}
    loss = -(p_mse + p_reg)
    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "mse": mse,
        "p_reg": p_reg,
        "reg": reg,
        "bayes_coeffs": mn,
        "coeffs": masked_coeffs,
        "alpha": prior[:-1],
        "beta": prior[-1],
        "tau": tau,
        "its": fwd_metric[0],
        "BIC": BIC_val,
    }

    return (
        loss,
        (
            (updated_model_state, updated_loss_state),
            metrics,
            (prediction, dt, theta, mn),
        ),
    )
