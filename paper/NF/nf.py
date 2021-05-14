import jax.numpy as jnp
from jax.scipy.stats import norm
from flax import linen as nn
from typing import Callable, Tuple, Sequence, List
from modax.models.networks import MLP
from jax.lax import scan
from jax import vmap


def planar_transform(z, params):
    u, w, b = params
    # making sure its invertible
    wu = jnp.dot(w, u.T)
    m_wu = -1 + jnp.log1p(jnp.exp(wu))
    u_hat = u + (m_wu - wu) * w / (jnp.dot(w, w.T) + 1e-6)
    # transforming
    a = jnp.tanh(jnp.dot(z, w.T) + b)
    f_z = z + a * u_hat
    psi = (1 - a ** 2) * w
    log_det = jnp.log(jnp.abs(1 + jnp.dot(psi, u_hat.T)))
    return f_z, log_det


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
        return planar_transform(inputs, (u, w, b))


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


class AmortizedNormalizingFlow(nn.Module):
    hyper_features: List[int]
    n_layers: int
    n_dims: int = 1

    def setup(self):
        n_out = (2 * self.n_dims + 1) * self.n_layers
        self.hyper_net = MLP(self.hyper_features + [n_out])

    def __call__(
        self, inputs: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, t = inputs
        params = self.reshape_params(self.hyper_net(t))
        z, log_jac = vmap(
            lambda batch_params, batch_x: scan(planar_transform, batch_x, batch_params)
        )(params, x)
        return norm.logpdf(z) + jnp.sum(log_jac, axis=-3)

    def reshape_params(self, params):
        u, w, b = jnp.split(
            params,
            [self.n_layers * self.n_dims, 2 * self.n_layers * self.n_dims],
            axis=1,
        )
        # u, w shape: (n_batch, n_layers, 1, n_dims)
        # b shape: (n_batch, n_layers, 1)
        u = u.reshape(-1, self.n_layers, 1, self.n_dims)
        w = w.reshape(-1, self.n_layers, 1, self.n_dims)
        b = b.reshape(-1, self.n_layers, 1)

        return u, w, b
