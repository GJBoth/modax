# %% Imports
from jax import random, jit, numpy as jnp
from typing import Sequence


from modax.data.burgers import burgers
from modax.training import mask_scheduler, create_stateful_update, Logger
from modax.layers import MaskedLeastSquares
from modax.feature_generators import library_backward
from modax.models.networks import MLP
from modax.losses.utils import mse
from modax.training.convergence import Convergence

from flax import optim, linen as nn
from flax.core import freeze
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import numpy as np


# %%
class CustomModel(nn.Module):
    """Simple feed-forward NN."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = MaskedLeastSquares()((dt, theta))
        return prediction, dt, theta, coeffs


def custom_loss_fn(params, state, model, x, y):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    MSE = mse(prediction, y)
    Reg = mse(dt.squeeze(), (theta @ coeffs).squeeze())
    loss = MSE + Reg
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs}

    return loss, (updated_state, metrics, (prediction, dt, theta, coeffs))


class CustomMaskEstimator:
    def __init__(self, threshold=0.1, *args, **kwargs):
        self.threshold = threshold
        self.reg = LassoCV(fit_intercept=False, *args, **kwargs)

    def fit(self, X, y):
        return self.reg.fit(np.array(X), np.array(y.squeeze())).coef_

    def __call__(self, X, y, threshold=0.1):
        X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)
        y_normed = y / jnp.linalg.norm(y, axis=0, keepdims=True)
        coeffs = self.fit(X_normed, y_normed)
        mask = np.abs(coeffs) > threshold
        return mask


# %% Making data
key = random.PRNGKey(42)

x = jnp.linspace(-3, 4, 50)
t = jnp.linspace(0.5, 5.0, 20)
t_grid, x_grid = jnp.meshgrid(t, x, indexing="ij")
u = burgers(x_grid, t_grid, 0.1, 1.0)

X = jnp.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
y = u.reshape(-1, 1)
y += 0.10 * jnp.std(y) * random.normal(key, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Building model and params
model = CustomModel([30, 30, 30, 1])
variables = model.init(key, X)

optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)

# %% Prepping some functions
custom_mask_fn = CustomMaskEstimator(threshold=0.1)

update = create_stateful_update(custom_loss_fn, model=model, x=X_train, y=y_train)
validation_metric = jit(
    lambda opt, state: custom_loss_fn(opt.target, state, model, X_test, y_test)[1][1]
)
logger = Logger()
scheduler = mask_scheduler()
converged = Convergence()
# %% Running
max_epochs = 1e4
for epoch in jnp.arange(max_epochs):
    (optimizer, state), train_metrics, output = update(optimizer, state)
    prediction, dt, theta, coeffs = output

    if epoch % 1000 == 0:
        print(f"Loss step {epoch}: {train_metrics['loss']}")

    if epoch % 25 == 0:
        val_metrics = validation_metric(optimizer, state)
        metrics = {
            **train_metrics,
            "val_mse": val_metrics["mse"],
            "val_reg": val_metrics["reg"],
        }
        logger.write(metrics, epoch)

        apply_sparsity, optimizer = scheduler(val_metrics["mse"], epoch, optimizer)

        if apply_sparsity:
            mask = custom_mask_fn(theta, dt)
            state = freeze({"vars": {"MaskedLeastSquares_0": {"mask": mask}}})

    if converged(epoch, coeffs):
        mask = custom_mask_fn(theta, dt)
        print(f"Converged at epoch {epoch} with mask {mask[:, None]}.")
        break

logger.close()
