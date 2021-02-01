# %% Imports
import jax
from jax import jit, numpy as jnp, lax, random
from modax.linear_model.bayesian_regression import bayesian_regression

from sklearn.linear_model import BayesianRidge


# %% Prepping data
data = jnp.load("test_data.npy", allow_pickle=True).item()
y, X = data["y"], data["X"]

X_normed = X / jnp.linalg.norm(X, axis=0)

# %% Baseline
reg = BayesianRidge(fit_intercept=False, compute_score=True)
reg.fit(X_normed, y.squeeze())

# %%


bayesian_regression(X_normed, y)
# %%
