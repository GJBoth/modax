from typing import Callable, Optional, Sequence, Tuple
from ..layers.feature_generators import library_backward, library_forward, library_paper
from ..layers.regression import LeastSquares, LeastSquaresMT
from .networks import MLP, MultiTaskMLP
from flax import linen as nn
import jax.numpy as jnp


class DeepmodBase(nn.Module):
    """Baseclass for DeepMoD models"""
    approx_fn: Callable
    approx_args: Sequence
    library_fn: Callable
    library_args: Sequence
    constraint_fn: Callable
    constraint_args: Sequence

    def setup(self):
        self.network = self.approx_fn(*self.approx_args)
        self.library = self.library_fn(*self.library_args)
        self.constraint = self.constraint_fn(*self.constraint_args)

    def __call__(self, inputs):
        prediction, features = self.library(self.network, inputs)
        coeffs = self.constraint(features)
        return prediction, *features, coeffs

def Deepmod(network_shape: Sequence[int], library_orders: Tuple[int, int]):
    return DeepmodBase(MLP, (network_shape, ), library_backward, (*library_orders, ), LeastSquares, ())

def DeepmodPaper(network_shape: Sequence[int]):
    return DeepmodBase(MLP, (network_shape, ), library_paper, (), LeastSquares, ())


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


class DeepmodGrad(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = LeastSquares()((dt, theta))
        # TODO: make sure eveyrwhere has (X, y) not (y, X)

        z = self.param("likelihood_params", self.likelihood_params_init, prediction, dt)
        return prediction, dt, theta, coeffs, z

    @staticmethod
    def likelihood_params_init(key, prediction, dt):
        z_mse = -jnp.log(jnp.var(prediction, axis=0))
        z_reg = -jnp.log(jnp.var(dt, axis=0))
        z = jnp.stack([z_mse, z_reg], axis=1)
        return z


