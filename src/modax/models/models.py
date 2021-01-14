from typing import Sequence, Callable
from ..feature_generators.feature_generators import library_backward, library_forward
from ..layers.regression import MaskedLeastSquares, LeastSquaresMT, LeastSquares
from .networks import MLP, MultiTaskMLP
from flax import linen as nn


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
    """Simple feed-forward NN."""

    features: Sequence[int]  # this is dataclass, so we dont use __init__
    noise_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = LeastSquares()((theta, dt))

        s = self.param("noise_mse", self.noise_init, (1,))
        t = self.param("noise_reg", self.noise_init, (1,))
        return prediction, dt, theta, coeffs, s, t
