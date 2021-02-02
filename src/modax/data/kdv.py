from jax import numpy as jnp


def doublesoliton(x, t, c, x0):
    # assert c[0] > c[1], "c1 has to be bigger than c[2]"

    xi0 = (
        jnp.sqrt(c[0]) / 2 * (x - c[0] * t - x0[0])
    )  # switch to moving coordinate frame
    xi1 = jnp.sqrt(c[1]) / 2 * (x - c[1] * t - x0[1])

    part_1 = 2 * (c[0] - c[1])
    numerator = c[0] * jnp.cosh(xi1) ** 2 + c[1] * jnp.sinh(xi0) ** 2
    denominator_1 = (jnp.sqrt(c[0]) - jnp.sqrt(c[1])) * jnp.cosh(xi0 + xi1)
    denominator_2 = (jnp.sqrt(c[0]) + jnp.sqrt(c[1])) * jnp.cosh(xi0 - xi1)
    u = part_1 * numerator / (denominator_1 + denominator_2) ** 2
    return u
