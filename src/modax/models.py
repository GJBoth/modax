from typing import Sequence
from modax.feature_generators import library_backward
from modax.layers import LeastSquares
from modax.networks import MLP
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

