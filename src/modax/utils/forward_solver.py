# %% Imports
import jax
from jax import jit, numpy as jnp, lax
from functools import partial


@partial(jit, static_argnums=(0, 2))
# / indicates the arguments before must be positional arguments and not keywerd
def fwd_solver(f, z_init, n_features=None, tol=1e-4, max_iter=300):
    # n_features calculates the norm over the first n_features of z.
    # Useful for when you're iterating over a but check your convergence on b
    # such as with SBL.
    def cond_fun(carry):
        _, _, (iteration, gap) = carry
        cond_norm = gap < tol
        cond_iter = iteration >= max_iter
        return ~jnp.logical_or(cond_norm, cond_iter)

    def body_fun(carry):
        z, z_prev, (iteration, gap) = carry
        gap = jnp.linalg.norm(z[:n_features] - z_prev[:n_features])
        return f(z), z, (iteration + 1, gap)

    init_carry = (
        f(z_init),
        z_init,
        (0, 10 * tol),
    )
    if n_features is None:
        n_features = z_init.shape[0]
    z_star, _, metrics = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star[n_features:], metrics


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0, 3))
def fixed_point_solver(f, args, z_init, n_features=None, tol=1e-4, max_iter=300):
    z_star, metrics = fwd_solver(
        lambda z: f(z, *args), z_init, n_features, tol=tol, max_iter=max_iter,
    )
    return z_star, metrics


@partial(jit, static_argnums=(0, 3))
def fixed_point_solver_fwd(f, args, z_init, n_features, tol, max_iter):
    z_star, metrics = fixed_point_solver(f, args, z_init, n_features, tol, max_iter)
    return (z_star, metrics), (z_star, tol, max_iter, args)


@partial(jit, static_argnums=(0,))
def fixed_point_solver_bwd(f, res, z_star_bar):
    z_star, tol, max_iter, args = res
    z_star_bar = z_star_bar[0]  # we dont take the gradient w.r.t metric

    _, vjp_a = jax.vjp(lambda args: f(z_star, *args), args)
    _, vjp_z = jax.vjp(lambda z: f(z, *args), z_star)
    res = vjp_a(
        fwd_solver(
            lambda u: vjp_z(u)[0] + z_star_bar,
            z_init=jnp.zeros_like(z_star),
            tol=1e-5,
            max_iter=500,
        )[0]
    )
    return (*res, None, None, None)  # None for init and to


fixed_point_solver.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)
