from typing import Sequence
from modax.feature_generators import library_backward, library_forward
from modax.layers import LeastSquares, LeastSquaresMT
from modax.networks import MLP, MultiTaskMLP
from flax import linen as nn


class Deepmod(nn.Module):
    """Simple feed-forward NN.
    """

    features: Sequence[int]  # this is dataclass, so we dont use __init__

    @nn.compact  # this function decorator lazily intializes the model, so it makes the layers the first time we call it
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        coeffs = LeastSquares()((theta, dt))
        return prediction, dt, theta, coeffs


class DeepmodMultiExp(nn.Module):
    """Simple feed-forward NN.
    """

    shared_features: Sequence[int]  # this is dataclass, so we dont use __init__
    specific_features: Sequence[int]
    n_tasks: int

    @nn.compact  # this function decorator lazily intializes the model, so it makes the layers the first time we call it
    def __call__(self, inputs):
        prediction, dt, theta = library_forward(
            MultiTaskMLP(self.shared_features, self.specific_features, self.n_tasks),
            inputs,
        )
        coeffs = LeastSquaresMT()((theta, dt))
        return prediction, dt, theta, coeffs
