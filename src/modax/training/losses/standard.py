from .utils import mse


def loss_fn_pinn(params, state, model, X, y, l=1.0):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs), updated_state = model.apply(
        variables, X, mutable=list(state.keys())
    )

    MSE = mse(prediction, y)
    Reg = mse(dt, theta @ coeffs)
    loss = MSE + l * Reg
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs}

    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_mse(params, state, model, X, y):
    return loss_fn_pinn(params, state, model, X, y, l=0.0)
