import jax
import jax.numpy as jnp

def siren_kernel_init(omega, is_first, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        fan_in = shape[0]
        if is_first:
            a = 1 / fan_in
        else:
            a = jnp.sqrt(6 / fan_in) / omega
        return jax.random.uniform(key, shape, dtype, minval=-a, maxval=a)
    return init