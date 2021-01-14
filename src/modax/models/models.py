from typing import Sequence
from ..feature_generators.feature_generators import library_backward, library_forward
from ..layers.regression import MaskedLeastSquares, LeastSquaresMT
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

