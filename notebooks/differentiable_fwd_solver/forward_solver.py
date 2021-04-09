# %% Imports
import jax
from jax import jit, numpy as jnp, lax
from functools import partial


@partial(jit, static_argnums=(0, 2))
def fwd_solver_jittable_diffable(f, z_init, break_fn):
    def cond_fun(carry):
        z_prev, z, iteration = carry
        with jax.disable_jit():
            conv = jnp.linalg.norm(z_prev - z) > 1e-5
        return conv

    def body_fun(carry):
        _, z, iteration = carry
        return z, f(z), iteration + 1

    carry = (
        z_init,
        f(z_init),
        1,
    )

    while cond_fun(carry):
        carry = body_fun(carry)
    _, z_star, metrics = carry
    return z_star, metrics


# Explicit method
@partial(jit, static_argnums=(0, 2))
def fwd_solver_diffable(f, z_init, break_fn, max_iterations):
    def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    def _cond_fun(carry):
        z_prev, z, iteration = carry
        if iteration < max_iterations:
            return break_fn(z_prev, z)
        else:
            return False

    def body_fun(carry):
        _, z, iteration = carry
        return z, f(z), iteration + 1

    init_carry = (
        z_init,
        f(z_init),
        1,
    )
    _, z_star, metrics = while_loop(_cond_fun, body_fun, init_carry)
    return z_star, metrics


# Implicit methods
@partial(jit, static_argnums=(0, 2))
def fwd_solver(f, z_init, cond_fun, max_iter=300):
    def _cond_fun(carry):
        z_prev, z, iteration = carry
        return jax.lax.cond(
            iteration >= max_iter,
            lambda _: False,
            lambda args: cond_fun(*args),
            (z_prev, z),
        )

    def body_fun(carry):
        _, z, iteration = carry
        return z, f(z), iteration + 1

    init_carry = (
        z_init,
        f(z_init),
        1,
    )
    _, z_star, metrics = lax.while_loop(_cond_fun, body_fun, init_carry)
    return z_star, metrics


# Explicit differentiation function.
@partial(jit, static_argnums=(0, 3))
def fixed_point_solver_explicit(f, args, z_init, cond_fun, max_iter):
    z_star, metrics = fwd_solver_diffable(
        lambda z: f(z, *args), z_init, cond_fun, max_iter
    )
    return z_star, metrics


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
@partial(jit, static_argnums=(0, 3))
def fixed_point_solver_implicit(f, args, z_init, cond_fun, max_iter=300):
    z_star, metrics = fwd_solver(
        lambda z: f(z, *args), z_init, cond_fun, max_iter=max_iter,
    )
    return z_star, metrics


def fixed_point_solver_fwd(f, args, z_init, cond_fun, max_iter):
    z_star, metrics = fixed_point_solver_implicit(f, args, z_init, cond_fun, max_iter)
    return (z_star, metrics), (z_star, args)


def fixed_point_solver_bwd(f, _, res, z_star_bar):
    z_star, args = res
    z_star_bar = z_star_bar[0]  # we dont take the gradient w.r.t metric

    _, vjp_a = jax.vjp(lambda args: f(z_star, *args), args)
    _, vjp_z = jax.vjp(lambda z: f(z, *args), z_star)
    res = vjp_a(
        fwd_solver(
            lambda u: vjp_z(u)[0] + z_star_bar,
            jnp.zeros_like(z_star),
            lambda z_prev, z: jnp.linalg.norm(z - z_prev) > 1e-5,
            max_iter=5000,
        )[0]
    )
    return (*res, None, None)  # None for init and max_iter


fixed_point_solver_implicit.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)
