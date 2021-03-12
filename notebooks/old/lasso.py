# %% Imports
import jax
from jax import jit, lax, numpy as jnp, random
from functools import partial
from sklearn.linear_model import Lasso

# %% Function declarations
@partial(jit, static_argnums=(0,))
def fwd_solver(f, z_init, tol=1e-4):
    def cond_fun(carry):
        z_prev, z = carry
        return (
            jnp.linalg.norm(z_prev[:-1] - z[:-1]) > tol
        )  # for numerical reasons, we check the change in alpha

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


def soft_threshold(x, a):
    return jnp.maximum(jnp.abs(x) - a, 0.0) * jnp.sign(x)


def lasso_ista_step(w, X, y, l, t):
    n_samples = 200
    df = -X.T @ (y - X @ w)
    return soft_threshold(w - t * df, t * l * n_samples)


def lasso(X, y, l, tol=1e-5):
    n_features = 12  # X.shape[-1]
    w_init = jnp.zeros((n_features, 1))
    X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)
    t = 1 / (jnp.linalg.norm(X_normed) ** 2)
    update = partial(lasso_ista_step, X=X_normed, y=y, l=l, t=t)

    return fwd_solver(update, w_init, tol=tol)


# %% Loading testdata
data = jnp.load("test_data.npy", allow_pickle=True).item()
y, X = data["y"], data["X"]

X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)

# %% Run it
lasso(X, y, l=1e-3)

# %% Baseline
reg = Lasso(fit_intercept=False, alpha=1e-3)
reg.fit(X_normed, y.squeeze()).coef_[:, None]

# %% Quick CV
key = random.PRNGKey(42)

n_samples, n_features = X.shape
n_folds = 5
idx = random.permutation(key, n_samples)

X_CV = jnp.stack(jnp.split(X[idx, :], n_folds), axis=0)
y_CV = jnp.stack(jnp.split(y[idx, :], n_folds), axis=0)

# %%
jax.vmap(lasso, in_axes=(0, 0, None), out_axes=(0))(X, y, 1e-3)

# %%

# %%
