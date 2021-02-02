import jax
import jax.numpy as jnp
from flax import linen as nn
from ..linear_model import ridge


class LeastSquares(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        y, X = inputs  # TODO: Write as ridge with l=0?
        mask = self.variable(
            "vars",
            "mask",
            lambda n_terms: jnp.ones((n_terms,), dtype=bool),
            X.shape[1],
        )

        X_masked = X * (~mask.value * 1e-6 + mask.value)
        coeffs = jnp.linalg.lstsq(X_masked, y)[0]

        return (
            coeffs * mask.value[:, None]
        )  # extra multiplication to compensate numerical errors


# LeastSquares = partialmethod(Ridge, l=0.0)


class Ridge(nn.Module):
    l: float = 1e-7

    @nn.compact
    def __call__(self, inputs):
        y, X = inputs
        mask = self.variable(
            "vars",
            "mask",
            lambda n_terms: jnp.ones((n_terms,), dtype=bool),
            X.shape[1],
        )
        coeffs = ridge(X * mask.value, y, l=self.l)[0]

        # extra multiplication to compensate numerical errors
        return coeffs * mask.value[:, None]


class LeastSquaresMT(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        theta, dt = inputs
        coeffs = self.fit(theta, dt)

        return coeffs

    def fit(self, X, y):
        return jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0), out_axes=0)(X, y)[0]
        # TODO: Merge with leastsquares?

