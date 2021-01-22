from jax import random
from losses.standard import loss_fn_pinn_stateful
from ..models.models import Deepmod
from flax import optim
from .training import train
from ..linear_model.mask_estimator import ThresholdedLasso


def deepmod(X, y):
    key = random.PRNGKey(42)
    model = Deepmod([50, 50, 1])
    variables = model.init(key, X)

    optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
    state, params = variables.pop("params")
    optimizer = optimizer.create(params)

    mask_fn = ThresholdedLasso(threshold=0.1)
    optimizer, state = train(
        model,
        optimizer,
        state,
        loss_fn_pinn_stateful,
        mask_fn,
        X,
        y,
        max_epochs=1e4,
        split=0.8,
        rand_seed=42,
    )
    return model, optimizer, state
