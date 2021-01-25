# %% Imports
import jax
from jax import jit, numpy as jnp, lax
from functools import partial


# Forward solver
@partial(jit, static_argnums=(0,))
def fwd_solver(f, z_init, tol=1e-4):
    def cond_fun(carry):
        z_prev, z = carry
        return (
            jnp.linalg.norm(z_prev[0] - z[0]) > tol
        )  # for numerical reasons, we check the change in alpha

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def fixed_point_solver(f, args, z_init, tol=1e-5):
    z_star = fwd_solver(lambda z: f(z, *args), z_init=z_init, tol=tol)
    return z_star


@partial(jit, static_argnums=(0,))
def fixed_point_solver_fwd(f, args, z_init, tol):
    z_star = fixed_point_solver(f, args, z_init, tol)
    return z_star, (z_star, tol, args)


@partial(jit, static_argnums=(0,))
def fixed_point_solver_bwd(f, res, z_star_bar):
    z_star, tol, args = res
    _, vjp_a = jax.vjp(lambda args: f(z_star, *args), args)
    _, vjp_z = jax.vjp(lambda z: f(z, *args), z_star)
    res = vjp_a(
        fwd_solver(
            lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(z_star), tol=tol
        )
    )
    return (*res, None, None)  # None for init and to


fixed_point_solver.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)
