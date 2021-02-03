# %% Imports
import jax
from jax import jit, numpy as jnp, lax, random
from modax.linear_model.bayesian_regression import bayesian_regression

from sklearn.linear_model import BayesianRidge


# %% Prepping data
data = jnp.load(
    "/home/gert-jan/Documents/modax/notebooks/bayesian_regression/using_svd/test_data.npy",
    allow_pickle=True,
).item()
y, X = data["y"], data["X"]

X_normed = X / jnp.linalg.norm(X, axis=0)

# %% Baseline
reg = BayesianRidge(fit_intercept=False, compute_score=True)
reg.fit(X_normed, y.squeeze())
print(reg.scores_[0], reg.coef_, reg.lambda_, reg.alpha_)
# %% No warm restart

loss, coeffs, prior, metrics = bayesian_regression(X_normed, y)
print(loss, coeffs, prior, metrics)
# %% Warm restart

prior_init = prior
loss, coeffs, prior, metrics = bayesian_regression(X_normed, y, prior_init)
print(loss, coeffs, prior, metrics)

