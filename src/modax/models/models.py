from typing import Callable, Sequence, Tuple

from jax._src.numpy.lax_numpy import zeros

from modax.losses.utils import precision
from ..feature_generators.feature_generators import library_backward, library_forward
from ..layers.regression import (
    MaskedLeastSquares,
    LeastSquaresMT,
    BayesianMaskedLeastSquares,
)
from .networks import MLP, MultiTaskMLP
from flax import linen as nn
import jax.numpy as jnp


class Deepmod(nn.Module):
    """Simple feed-forward NN."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = MaskedLeastSquares()((dt, theta))
        return prediction, dt, theta, coeffs


class DeepmodMultiExp(nn.Module):
    """Simple feed-forward NN."""

    shared_features: Sequence[int]
    specific_features: Sequence[int]
    n_tasks: int

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_forward(
            MultiTaskMLP(self.shared_features, self.specific_features, self.n_tasks),
            inputs,
        )
        coeffs = LeastSquaresMT()((theta, dt))
        return prediction, dt, theta, coeffs


class DeepmodBayes(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = MaskedLeastSquares()((dt, theta))

        z = self.param("likelihood_params", self.likelihood_params_init, prediction, dt)
        return prediction, dt, theta, coeffs, z

    @staticmethod
    def likelihood_params_init(key, prediction, dt):
        z_mse = -jnp.log(jnp.var(prediction, axis=0))
        z_reg = -jnp.log(jnp.var(dt, axis=0))
        z = jnp.stack([z_mse, z_reg], axis=1)
        return z


"""
class DeepmodBayesPrecalc(nn.Module):
    features: Sequence[int]
    prior_params_mse: Tuple[float]
    prior_params_reg: Tuple[float]

    @nn.compact
    def __call__(self, inputs):
        initialized = self.has_variable("vars", "precision")

        X, y = inputs
        prediction, dt, theta = library_backward(MLP(self.features), X)
        coeffs = MaskedLeastSquares()((dt, theta))

        z = self.variable(
            "vars", "precision", lambda n_terms: jnp.zeros((1, n_terms)), 2
        )
        training = self.variable("vars", "training", lambda: True)

        if initialized and training.val is True:
            tau = precision(y, prediction, *self.prior_params_mse)
            nu = precision(dt, theta @ coeffs, *self.prior_params_reg)
            z = jnp.stack([tau, nu])

        return prediction, dt, theta, coeffs, z

"""


class DeepmodBayesNew(nn.Module):
    features: Sequence[int]
    prior_params_reg: Tuple[float]

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        p_reg, (coeffs, reg) = BayesianMaskedLeastSquares(self.prior_params_reg)(
            (dt, theta)
        )

        return prediction, p_reg, (reg, coeffs)

