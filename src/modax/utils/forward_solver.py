# %% Imports
import jax
from jax import jit, numpy as jnp, lax
from functools import partial


@partial(jit, static_argnums=(0,))
def fwd_solver(f, z_init, tol=1e-4, max_iter=300):
    def cond_fun(carry):
        _, _, (iteration, gap) = carry
        cond_norm = gap < tol
        cond_iter = iteration >= max_iter
        return ~jnp.logical_or(cond_norm, cond_iter)

    def body_fun(carry):
        z, z_prev, (iteration, gap) = carry
        gap = jnp.linalg.norm(z[:-1] - z_prev[:-1])
        return f(z), z, (iteration + 1, gap)

    init_carry = (
        f(z_init),
        z_init,
        (0, 10 * tol),
    )
    z_star, _, metrics = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star, metrics


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def fixed_point_solver(f, args, z_init, tol=1e-4, max_iter=300):
    z_star, metrics = fwd_solver(
        lambda z: f(z, *args), z_init=z_init, tol=tol, max_iter=max_iter
    )
    return z_star, metrics


@partial(jit, static_argnums=(0,))
def fixed_point_solver_fwd(f, args, z_init, tol, max_iter):
    z_star, metrics = fixed_point_solver(f, args, z_init, tol, max_iter)
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
            tol=tol,
            max_iter=max_iter,
        )[0]
    )
    return (*res, None, None, None)  # None for init and to


fixed_point_solver.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)
