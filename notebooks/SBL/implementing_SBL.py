# %% Imports
import jax
from jax import jit, numpy as jnp
from modax.linear_model.SBL import SBL

from sklearn.linear_model import ARDRegression

# %% Prepping data
data = jnp.load(
    "/home/gert-jan/Documents/modax/notebooks/SBL/test_data.npy", allow_pickle=True
).item()
y, X = data["y"], data["X"]

X = X / jnp.linalg.norm(X, axis=0)
y = y.squeeze()
# %% Baseline
reg = ARDRegression(fit_intercept=False, compute_score=True, threshold_lambda=1e8)
reg.fit(X, y.squeeze())


# %%
prior, metric = SBL(X, y)
print(prior, metric)
print(reg.lambda_, reg.alpha_)
# %%
