# %% Imports
from jax import random, numpy as jnp, jit
from modax.data.burgers import burgers
from modax.training import train_early_stop
from modax.models import Deepmod
from modax.training.losses import loss_fn_pinn, loss_fn_mse
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
update_fn = create_update(loss_fn_pinn, (model, X_train, y_train, 1.0))

# Validation loss is mse on testset.
val_fn = jit(
    lambda opt, state: loss_fn_mse(opt.target, state, model, X_test, y_test)[0]
)
# %%
optimizer, state = train_early_stop(
    update_fn, val_fn, optimizer, state, max_epochs=10000, delta=0.0, patience=2000
)

# %%
