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

n_samples, n_features = X.shape

alpha_ = reg.alpha_  # 1.0 / (jnp.var(y) + 1e-7)
lambda_ = reg.lambda_  # jnp.ones((n_features,))
keep_lambda = reg.lambda_ < 1e4
coef_ = reg.coef_

lambda_prior = (1e-6, 1e-6)
alpha_prior = (1e-6, 1e-6)


# %%
def update_sigma(X, alpha_, lambda_, keep_lambda):
    X_keep = X[:, keep_lambda]
    gram = jnp.dot(X_keep.T, X_keep)
    eye = jnp.eye(gram.shape[0])
    sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
    sigma_ = jnp.linalg.pinv(sigma_inv)
    return sigma_


def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
    coef_ = jax.ops.index_update(
        coef_,
        keep_lambda,
        alpha_ * jnp.linalg.multi_dot([sigma_, X[:, keep_lambda].T, y]),
    )
    return coef_


# %%
alpha_ = 1.0 / (jnp.var(y) + 1e-7)
lambda_ = jnp.ones((n_features,))
keep_lambda = jnp.ones((n_features,), dtype=bool)
coef_ = jnp.zeros_like(lambda_)

lambda_prior = (1e-6, 1e-6)
alpha_prior = (1e-6, 1e-6)
threshold = 1e6

for iter_ in jnp.arange(500):
    sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
    coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)

    # Update alpha and lambda
    rmse_ = jnp.sum((y - jnp.dot(X, coef_)) ** 2)
    gamma_ = 1.0 - lambda_[keep_lambda] * jnp.diag(sigma_)
    lambda_ = jax.ops.index_update(
        lambda_,
        keep_lambda,
        (gamma_ + 2.0 * lambda_prior[0])
        / ((coef_[keep_lambda]) ** 2 + 2.0 * lambda_prior[1]),
    )

    alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_prior[0]) / (
        rmse_ + 2.0 * alpha_prior[1]
    )

    # Prune the weights with a precision over a threshold
    keep_lambda = lambda_ < threshold
    coef_ = jax.ops.index_update(coef_, ~keep_lambda, 0)

# %%
