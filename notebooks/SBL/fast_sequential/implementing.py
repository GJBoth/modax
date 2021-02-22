# %% imports
from jax import random
import jax.numpy as jnp
import jax
from jax.ops import index, index_add, index_update
from sklearn.linear_model import ARDRegression

# %% loading test data
data = jnp.load("test_data.npy", allow_pickle=True).item()
y, X = data["y"], data["X"]

# %% Initialization of priors
n_samples, n_features = X.shape
alpha = jnp.zeros((n_features,))
beta = 1 / jnp.var(y) * 0.1

X = X / jnp.linalg.norm(X, axis=0)  # norming X, makes eq. easier
t = y

# %% Baseline
reg = ARDRegression(fit_intercept=False)
reg.fit(X, t.squeeze())
print(reg.lambda_)

# %% First iteration; finding best intial guess
sigma_noise = jnp.var(t) * 0.1
alpha, features = jnp.zeros((n_features,)), jnp.zeros((n_features,), dtype=bool)


normed_projection = (jnp.dot(X.T, y) ** 2).squeeze() / jnp.linalg.norm(X, axis=0) ** 2
idx = jnp.argmax(normed_projection)

# adding idx to active features
features = index_update(features, idx, 1)

# Updating alpha
ini_alpha = jnp.linalg.norm(X[:, idx]) ** 2 / (normed_projection[idx] - sigma_noise)
alpha = index_update(alpha, idx, ini_alpha)

# %% Calculating first sigma and mu
def update_sigma(Theta, alpha, beta):
    sigma_inv = jnp.diag(alpha) + beta * Theta.T @ Theta
    return jnp.linalg.pinv(sigma_inv)


def update_mu(Theta, t, beta, Sigma):
    mu = beta * Sigma @ Theta.T @ t
    return mu


Theta = X[:, features]
alpha_active = alpha[features]
Sigma = update_sigma(Theta, alpha_active, beta)
mu = update_mu(Theta, t, beta, Sigma)

# %% Calculating S and Q for all
def Sm(phi_m, Theta, Sigma, sigma_noise):
    n_samples, _ = Theta.shape
    B = 1 / sigma_noise * jnp.eye(n_samples)
    Sm = phi_m.T @ B @ phi_m
    Sm -= phi_m.T @ B @ Theta @ Sigma @ Theta.T @ B @ phi_m
    return Sm.squeeze()


def Qm(phi_m, Theta, t, Sigma, sigma_noise):
    n_samples, _ = Theta.shape
    B = 1 / sigma_noise * jnp.eye(n_samples)
    Qm = phi_m.T @ B @ t
    Qm -= phi_m.T @ B @ Theta @ Sigma @ Theta.T @ B @ t
    return Qm.squeeze()


big_S = jnp.stack(
    [Sm(X[:, [idx]], Theta, Sigma, sigma_noise) for idx in jnp.arange(n_features)],
    axis=0,
)
big_Q = jnp.stack(
    [Qm(X[:, [idx]], Theta, t, Sigma, sigma_noise) for idx in jnp.arange(n_features)],
    axis=0,
)


def sm(big_S, alpha, features):
    small_s_in_model = (
        alpha[features] * big_S[features] / (alpha[features] - big_S[features])
    )
    small_s = index_update(big_S, features, small_s_in_model)
    return small_s


def qm(big_Q, big_S, alpha, features):
    small_q_in_model = (
        alpha[features] * big_Q[features] / (alpha[features] - big_S[features])
    )
    small_Q = index_update(big_Q, features, small_q_in_model)
    return small_Q


small_s = sm(big_S, alpha, features)
small_q = qm(big_Q, big_S, alpha, features)

# %% Updating alpha


def update_features(small_q, small_s, idx, alpha, features):
    theta_i = (small_q ** 2 - small_s)[idx]
    if theta_i >= 0:
        if features[idx] == 0:
            features = index_update(features, idx, 1)

        updated_alpha = small_s[idx] ** 2 / theta_i
        alpha = index_update(alpha, idx, updated_alpha)
    else:
        features = index_update(features, idx, 0)
        alpha = index_update(alpha, idx, 0)
    return alpha, features


key = jax.random.PRNGKey(42)
idx = jax.random.choice(key, n_features)
alpha, features = update_features(small_q, small_s, idx, alpha, features)

# %% Updating sigma and mu

Theta = X[:, features]
alpha_active = alpha[features]
Sigma = update_sigma(Theta, alpha_active, beta)
mu = update_mu(Theta, t, beta, Sigma)
sigma_noise = jnp.sum((t - Theta @ mu) ** 2) / (
    n_samples - n_features + jnp.sum(alpha_active * jnp.diag(Sigma))
)


# %% Running it all in a loop


max_its = 500
for _ in jnp.arange(max_its):
    key, _ = random.split(key, 2)
    idx = jax.random.choice(key, n_features)
    big_S = jnp.stack(
        [Sm(X[:, [idx]], Theta, Sigma, sigma_noise) for idx in jnp.arange(n_features)],
        axis=0,
    )
    big_Q = jnp.stack(
        [
            Qm(X[:, [idx]], Theta, t, Sigma, sigma_noise)
            for idx in jnp.arange(n_features)
        ],
        axis=0,
    )

    small_s = sm(big_S, alpha, features)
    small_q = qm(big_Q, big_S, alpha, features)

    alpha, features = update_features(small_q, small_s, idx, alpha, features)

    Theta = X[:, features]
    alpha_active = alpha[features]
    Sigma = update_sigma(Theta, alpha_active, beta)
    mu = update_mu(Theta, t, beta, Sigma)
    sigma_noise = jnp.sum((t - Theta @ mu) ** 2) / (
        n_samples - n_features + jnp.sum(alpha_active * jnp.diag(Sigma))
    )
# %% Running it all in a loop
# %%
