# %% Imports
from jax import random, numpy as jnp, jit

from modax.data.burgers import burgers
from modax.models import Deepmod
from modax.training.losses import loss_fn_pinn, loss_fn_mse
from modax.linear_model.mask_estimator import ThresholdedLasso

from modax.training import train_full
from modax.training.utils import create_update
from sklearn.model_selection import train_test_split
from flax import optim

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
model = Deepmod([30, 30, 30, 1])
variables = model.init(key, X)

optimizer = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
state, params = variables.pop("params")
optimizer = optimizer.create(params)


# Mask update function is lasso, validation loss is mse on testset.
update_fn = create_update(loss_fn_pinn, (model, X_train, y_train, 1.0))
mask_update_fn = ThresholdedLasso(0.1)
val_fn = jit(
    lambda opt, state: loss_fn_mse(opt.target, state, model, X_test, y_test)[0]
)
# %% Runnig
optimizer, state = train_full(
    update_fn,
    val_fn,
    mask_update_fn,
    optimizer,
    state,
    max_epochs=10000,
    convergence_args={"patience": 200, "delta": 1e-3},
    mask_update_args={"patience": 500, "delta": 1e-5, "periodicity": 200},
)

# %%
