# %% Imports
import jax
from jax import jit, numpy as jnp

from sklearn.linear_model import ARDRegression

# %% Prepping data
data = jnp.load(
    "/home/gert-jan/Documents/modax/notebooks/SBL/test_data.npy", allow_pickle=True
).item()
y, X = data["y"], data["X"]

X = X / jnp.linalg.norm(X, axis=0)
y = y.squeeze()
# %% Baseline
reg = ARDRegression(fit_intercept=False, compute_score=True, threshold_lambda=1e6)
reg.fit(X, y.squeeze())


# %%
@jit
def update_sigma(gram, alpha_, lambda_):
    sigma_inv = lambda_ * jnp.eye(gram.shape[0]) + alpha_ * gram
    L_inv = jnp.linalg.pinv(jnp.linalg.cholesky(sigma_inv))
    sigma_ = jnp.dot(L_inv.T, L_inv)
    return sigma_


@jit
def update_coeff(XT_y, alpha_, sigma_):
    coef_ = alpha_ * jnp.linalg.multi_dot([sigma_, XT_y])
    return coef_


@jit
def update(alpha_, lambda_, X, y, gram, XT_y, alpha_prior, lambda_prior):
    n_samples = X.shape[0]
    sigma_ = update_sigma(gram, alpha_, lambda_)
    coef_ = update_coeff(XT_y, alpha_, sigma_)

    # Update alpha and lambda
    rmse_ = jnp.sum((y - jnp.dot(X, coef_)) ** 2)
    gamma_ = 1.0 - lambda_ * jnp.diag(sigma_)
    lambda_ = (gamma_ + 2.0 * lambda_prior[0]) / ((coef_ ** 2 + 2.0 * lambda_prior[1]))

    alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_prior[0]) / (
        rmse_ + 2.0 * alpha_prior[1]
    )

    return alpha_, lambda_


# %%
n_samples, n_features = X.shape
alpha_ = 1.0 / (jnp.var(y) + 1e-7)
lambda_ = jnp.ones((n_features,))

lambda_prior = (1e-6, 1e-6)
alpha_prior = (1e-6, 1e-6)

gram = jnp.dot(X.T, X)
XT_y = jnp.dot(X.T, y)

for iter_ in jnp.arange(100):
    alpha_, lambda_ = update(
        alpha_, lambda_, X, y, gram, XT_y, alpha_prior, lambda_prior
    )
# %%
