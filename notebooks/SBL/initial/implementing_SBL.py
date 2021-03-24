# %% Imports
import jax
from jax import jit, numpy as jnp
from jax.api import value_and_grad
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


# %% Cold restart
loss, mn, prior, metric = SBL(X, y)
print(loss, mn, prior, metric)

# %% Warm restart
prior_init = prior
loss, mn, prior, metric = SBL(X, y, prior_init)
print(loss, mn, prior, metric)
print(reg.scores_[-1], reg.coef_, reg.lambda_, reg.alpha_)

# %% Grad
print(jax.grad(lambda x: SBL(x, y, prior_init)[0])(X))

# %%
grad_fn = jax.value_and_grad(lambda x, y: SBL(x, y, prior_init)[0])
# %%
