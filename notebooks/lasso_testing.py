from jax import numpy as jnp, random
from .lasso import fwd_solver, lasso

# %% Loading testdata
data = jnp.load("test_data.npy", allow_pickle=True).item()
y, X = data["y"], data["X"]

X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)
n_samples, n_features = X.shape
w_init = jnp.zeros((n_features, 1))
# %% Run it
lasso(X, y, w_init, l=(1e-3 * n_samples))

# %% Baseline
reg = Lasso(fit_intercept=False, alpha=1e-3)
reg.fit(X_normed, y.squeeze()).coef_[:, None]

# %% Quick CV
key = random.PRNGKey(42)

n_samples, n_features = X.shape
n_folds = 5
idx = random.permutation(key, n_samples)

X_CV = jnp.stack(jnp.split(X_normed[idx, :], n_folds), axis=0)
y_CV = jnp.stack(jnp.split(y[idx, :], n_folds), axis=0)
w_init = jnp.zeros((n_folds, n_features, 1))

# %%
jax.vmap(CVlasso, in_axes=(0, 0, None), out_axes=(0))(X_CV, y_CV, 1e-3 * n_samples / 5)

# %%


# %%
jnp.linalg.norm(X_CV)
# %%
jnp.linalg.norm(X_normed)
# %%
