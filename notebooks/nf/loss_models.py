import jax.numpy as jnp
from modax.models import DeepmodBase
from modax.layers.feature_generators import library_backward
from modax.layers.regression import LeastSquares
from typing import Sequence, Tuple
from nf import AmortizedNormalizingFlow



def DeepmodNF(
    network_shape: Sequence[int], n_flow_layers: int, library_orders: Tuple[int, int]
):
    return DeepmodBase(
        AmortizedNormalizingFlow,
        (network_shape, n_flow_layers,),
        library_backward,
        (*library_orders,),
        LeastSquares,
        (),
    )

def loss_fn_SBL(params, state, model, x):
    model_state, loss_state = state
    n_samples = x[0].size
    beta_prior = (n_samples, n_samples * 1e-5)
    
    variables = {"params": params, **model_state}
    (log_p, dt, theta, coeffs), updated_model_state = model.apply(variables, x, mutable=list(model_state.keys()))
    p_mse = jnp.sum(log_p)

    prior_init = loss_state["prior_init"]
    p_reg, mn, prior, fwd_metric = SBL(
        theta,
        dt,
        prior_init=prior_init,
        hyper_prior=((1e-6, 1e-6), beta_prior),
        tol=1e-4,
        max_iter=2000,
    )
    reg = jnp.mean((dt - jnp.dot(theta, coeffs)) ** 2)
    updated_loss_state = {"prior_init": prior}
    loss = -(p_mse + p_reg)
    metrics = {
        "loss": loss,
        "p_mse": p_mse,
        "p_reg": p_reg,
        "coeffs": coeffs,
        "reg": reg,
        "bayes_coeffs": mn,
        "alpha": prior[:-1],
        "beta": prior[-1],
        "its": fwd_metric[0],
    }

    return (
        loss,
        (
            (updated_model_state, updated_loss_state),
            metrics,
            (log_p, dt, theta, mn),
        ),
    )

def NF_library(poly_order=2):
    def library_fn(model, inputs):
        X, t = inputs
        f = lambda x, t: jnp.exp(model((x, t)))

        # prediction
        pred = model(inputs).reshape(-1, 1)
        polynomials = poly_fn(jnp.exp(pred))
        # time deriv
        df_t = vgrad_forward(lambda t: f(X, t), t, input_idx=0).reshape(-1, 1)

        # spatial derivs
        df_x = partial(vgrad_backward, lambda x: f(x, t))
        d2f_x = partial(vgrad_backward, df_x)
        d3f_x = partial(vgrad_backward, d2f_x)
        derivs = jnp.concatenate([df_x(X).reshape(-1, 1), d2f_x(X).reshape(-1, 1), d3f_x(X).reshape(-1, 1)], axis=1)
        
        # library
        u = jnp.concatenate([jnp.ones_like(pred), polynomials], axis=1)[:, :, None]
        du = jnp.concatenate([jnp.ones_like(pred), derivs], axis=1)[:, None, :]
        n_features = 4 * (poly_order + 1)
        theta = jnp.matmul(u, du).reshape(-1, n_features)
        return pred, (df_t, theta)
    poly_fn = partial(nth_polynomial, order=poly_order)
    return library_fn