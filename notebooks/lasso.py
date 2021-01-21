# %% Imports
from jax import jit, lax, numpy as jnp
from functools import partial

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


@jit
def soft_threshold(x, a):
    return jnp.maximum(jnp.abs(x) - a, 0.0) * jnp.sign(x)


@jit
def lasso_ista_step(w, X, y, l, t):
    n_samples = X.shape[-2]
    df = -jnp.matmul(X.T, (y - jnp.matmul(X, w)))
    return soft_threshold(w - t * df, t * l * n_samples)


@jit
def lasso(X, y, l, tol=1e-5):
    t = 1 / (jnp.linalg.norm(X) ** 2)
    w_init = jnp.zeros((X.shape[1],))
    update = partial(lasso_ista_step, X=X, y=y, l=l, t=t)

    return fwd_solver(update, w_init, tol=tol)

