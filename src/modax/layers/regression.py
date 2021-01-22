import jax
import jax.numpy as jnp
from flax import linen as nn
from ..linear_model import ridge
from typing import Tuple


class LeastSquares(nn.Module):
    def __call__(self, inputs):
        theta, dt = inputs
        coeffs = self.fit(theta, dt)

        return coeffs

    def fit(self, X, y):
        return jnp.linalg.lstsq(X, y)[0]


class LeastSquaresMT(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        theta, dt = inputs
        coeffs = self.fit(theta, dt)

        return coeffs

    def fit(self, X, y):
        return jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0), out_axes=0)(X, y)[0]


class MaskedLeastSquares(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        y, X = inputs
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


class Ridge(nn.Module):
    l: float = 1e-7

    @nn.compact
    def __call__(self, inputs):
        y, X = inputs
        mask = self.variable(
            "mask",
            "active terms",
            lambda n_terms: jnp.ones((n_terms,), dtype=bool),
            X.shape[1],
        )

        coeffs = ridge(X * mask.value, y, l=self.l)

        return (
            coeffs * mask.value[:, None]
        )  # extra multiplication to compensate numerical errors


class BayesianMaskedLeastSquares(nn.Module):
    prior_params: Tuple[float]

    @nn.compact
    def __call__(self, inputs):
        is_initialized = self.has_variable("vars", "reg_precision")
        y, X = inputs
        mask = self.variable(
            "vars",
            "mask",
            lambda n_terms: jnp.ones((n_terms,), dtype=bool),
            X.shape[1],
        )

        X_masked = X * (~mask.value * 1e-6 + mask.value)
        coeffs, sq_res, _, _ = jnp.linalg.lstsq(X_masked, y)

        # Calculating coeffs
        coeffs *= mask.value[:, None]  # round off numerical errors
        mse = sq_res / sq_res.shape[0]

        # Calculating precision
        nu = self.variable("vars", "reg_precision", lambda: jnp.ones((1,)),)
        if is_initialized:
            nu.val = precision(mse, *self.prior_params)

        # Loss
        p_reg = normal_LL(mse, nu)
        p_reg += gamma_LL(nu, *self.prior_params)

        return p_reg, (coeffs, mse)


def precision(mse, alpha, beta):
    # calculates precision parameter with a gamma prior
    n_samples = mse.shape[0]
    tau = (n_samples + 2 * (alpha - 1)) / (n_samples * mse + 2 * beta)
    return tau


def normal_LL(mse, tau):
    # tau = 1 / sigma**2, for numerical reasons.
    n_samples = mse.shape[0]

    # 2 before tau to compensate for 1/2
    p = -0.5 * n_samples * (tau * mse - jnp.log(tau) + jnp.log(2 * jnp.pi))

    return p


def gamma_LL(x, alpha, beta):
    # log pdf of gamma dist, dropped constant factors since it can be improper.
    p = (alpha - 1) * jnp.log(x) - beta * x
    return p
