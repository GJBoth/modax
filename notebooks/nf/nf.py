import jax.numpy as jnp
from jax.scipy.stats import norm
from flax import linen as nn
from typing import Callable, Tuple


def planar_transform(
    u: jnp.ndarray, w: jnp.ndarray, b: jnp.array, z: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
    wu = w @ u.T
    m_wu = -1 + jnp.log1p(jnp.exp(wu))
    u_hat = u + (m_wu - wu) * w / (w @ w.T)

    # Calculate transformed coordinates
    f_z = z + u_hat * jnp.tanh(z @ w.T + b)

    # Determining absolute log Jacobian
    psi = (1 - jnp.tanh(z @ w.T + b) ** 2) @ w
    abs_det = jnp.abs(1 + psi @ u_hat.T)
    log_J = jnp.log(abs_det)

    return f_z, log_J


class PlanarTransform(nn.Module):
    """Flax module for planar transform func."""

    u_init: Callable = nn.initializers.lecun_normal()
    w_init: Callable = nn.initializers.lecun_normal()
    b_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n_dims = inputs.shape[1]
        u = self.param("u", self.u_init, (1, n_dims))
        w = self.param("w", self.w_init, (1, n_dims))
        b = self.param("b", self.b_init, (1,))
        return planar_transform(u, w, b, inputs)


class NormalizingFlow(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        log_jacob = jnp.zeros((inputs.shape[0], 1))
        z = inputs
        for _ in jnp.arange(self.n_layers):
            z, layer_log_jacob = PlanarTransform()(z)
            log_jacob += layer_log_jacob
        log_p = norm.logpdf(z) + log_jacob
        return log_p
