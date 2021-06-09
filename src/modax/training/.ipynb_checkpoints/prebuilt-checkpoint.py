from jax import random
from modax.losses.standard import loss_fn_pinn_stateful
from ..models.models import Deepmod
from flax import optim
from .utils import mask_updater
from sklearn.linear_model import LassoCV
from .training import train


def deepmod(X, y):
    key = random.PGNRKey(42)
    model = Deepmod([50, 50, 1])
    variables = model.init(key, X)

    optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
    state, params = variables.pop("params")
    optimizer = optimizer.create(params)

    update_mask = lambda X, y: mask_updater(
        X, y, LassoCV(fit_intercept=False), threshold=0.1
    )
    optimizer, state = train(
        model,
        optimizer,
        state,
        loss_fn_pinn_stateful,
        update_mask,
        X,
        y,
        max_epochs=1e4,
        split=0.8,
        rand_seed=42,
    )
    return model, optimizer, state
