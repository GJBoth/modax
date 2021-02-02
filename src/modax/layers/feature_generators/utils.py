import jax
from jax.ops import index_update, index
import jax.numpy as jnp


def vgrad_backward(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


def vgrad_forward(f, x, input_idx):
    s = index_update(jnp.zeros_like(x), index[:, input_idx], 1)
    _, jvp = jax.jvp(f, (x,), (s,))
    return jvp
