from .utils import mse


def loss_fn_mse(params, model, x, y):
    """first argument should always be params!"""
    prediction = model.apply(params, x)[0]
    loss = mse(prediction, y)
    metrics = {"loss": loss, "mse": loss}
    return loss, metrics


def loss_fn_pinn(params, model, x, y):
    prediction, dt, theta, coeffs = model.apply(params, x)

    MSE = mse(prediction, y)
    Reg = mse(dt.squeeze(), (theta @ coeffs).squeeze())
    loss = MSE + Reg
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs}

    return loss, metrics


def loss_fn_pinn_stateful(params, state, model, x, y):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    MSE = mse(prediction, y)
    Reg = mse(dt.squeeze(), (theta @ coeffs).squeeze())
    loss = MSE + Reg
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs}

    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


def loss_fn_pinn_multi(params, model, x, y):
    prediction, dt, theta, coeffs = model.apply(params, x)
    loss = mse(prediction, y) + mse(dt.squeeze().T, (theta @ coeffs).squeeze().T)
    return loss
