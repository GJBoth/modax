from typing import Callable
import jax.numpy as jnp
from jax import random


def random_walk(key, initial_condition: Callable, n_steps, n_walkers, D, v, dt):
    """Generates random walk."""
    # Normal is location scale family
    steps = v * dt + jnp.sqrt(2 * D * dt) * \
        random.normal(key, shape=(n_steps, n_walkers))
    steps = jnp.concatenate(
        (initial_condition(key, (1, n_walkers)), steps), axis=0)
    locs = jnp.cumsum(steps, axis=0)
    t = jnp.arange(n_steps + 1) * dt

    return locs, t
