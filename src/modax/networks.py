from typing import Sequence
from jax import numpy as jnp
from flax import linen as nn

from modax.layers import MultiTaskDense


class MLP(nn.Module):
    """Simple feed-forward NN.
    """

    features: Sequence[int]  # this is dataclass, so we dont use __init__

    @nn.compact  # this function decorator lazily intializes the model, so it makes the layers the first time we call it
    def __call__(self, inputs):
        x = inputs  # we overwrite x so we copy it to a new tensor
        for feature in self.features[:-1]:
            x = nn.tanh(nn.Dense(feature)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class MultiTaskMLP(nn.Module):
    """Simple feed-forward NN.
    """

    shared_features: Sequence[int]  # this is dataclass, so we dont use __init__
    specific_features: Sequence[int]
    n_tasks: int

    @nn.compact  # this function decorator lazily intializes the model, so it makes the layers the first time we call it
    def __call__(self, inputs):
        x = inputs  # we overwrite x so we copy it to a new tensor
        for feature in self.shared_features:
            x = nn.tanh(nn.Dense(feature)(x))
        x = jnp.repeat(
            jnp.expand_dims(x, axis=0), repeats=self.n_tasks, axis=0
        )  # If we batch, can we do without copying data?
        for feature in self.specific_features[:-1]:
            x = nn.tanh(MultiTaskDense(feature, self.n_tasks)(x))
        x = MultiTaskDense(self.specific_features[-1], self.n_tasks)(x)
        return x
