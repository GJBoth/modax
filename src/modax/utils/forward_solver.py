# %% Imports
import jax
from jax import jit, numpy as jnp, lax
from functools import partial


@partial(jit, static_argnums=(0,))
def fwd_solver(f, z_init, tol=1e-4, max_iter=300):
    def cond_fun(carry):
        iteration, z_prev, z = carry
        # we check the change in alpha (element 0 in z tuple)
        # and the maximum number of iterations
        cond_norm = jnp.linalg.norm(z_prev[:-1] - z[:-1]) < tol
        cond_iter = iteration >= max_iter
        return ~jnp.logical_or(cond_norm, cond_iter)

    def body_fun(carry):
        iteration, _, z = carry
        return iteration + 1, z, f(z)

    init_carry = (0, z_init, f(z_init))  # first arg is iteration count
    _, _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def fixed_point_solver(f, args, z_init, tol=1e-5, max_iter=300):
    z_star = fwd_solver(
        lambda z: f(z, *args), z_init=z_init, tol=tol, max_iter=max_iter
    )
    return z_star


@partial(jit, static_argnums=(0,))
def fixed_point_solver_fwd(f, args, z_init, tol, max_iter):
    z_star = fixed_point_solver(f, args, z_init, tol, max_iter)
    return z_star, (z_star, tol, max_iter, args)


@partial(jit, static_argnums=(0,))
def fixed_point_solver_bwd(f, res, z_star_bar):
    z_star, tol, max_iter, args = res
    _, vjp_a = jax.vjp(lambda args: f(z_star, *args), args)
    _, vjp_z = jax.vjp(lambda z: f(z, *args), z_star)
    res = vjp_a(
        fwd_solver(
            lambda u: vjp_z(u)[0] + z_star_bar,
            z_init=jnp.zeros_like(z_star),
            tol=tol,
            max_iter=max_iter,
        )
    )
    return (*res, None, None, None)  # None for init and to


fixed_point_solver.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)
