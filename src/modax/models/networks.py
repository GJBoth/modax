from typing import Sequence
from jax import numpy as jnp
from flax import linen as nn

from ..layers.network import MultiTaskDense, SineLayer


class MLP(nn.Module):
    """Simple feed-forward NN."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for feature in self.features[:-1]:
            x = nn.tanh(nn.Dense(feature)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class MultiTaskMLP(nn.Module):
    """Simple feed-forward NN."""

    shared_features: Sequence[int]
    specific_features: Sequence[int]
    n_tasks: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for feature in self.shared_features:
            x = nn.tanh(nn.Dense(feature)(x))
        x = jnp.repeat(
            jnp.expand_dims(x, axis=0), repeats=self.n_tasks, axis=0
        )  # If we batch, can we do without copying data?
        for feature in self.specific_features[:-1]:
            x = nn.tanh(MultiTaskDense(feature, self.n_tasks)(x))
        x = MultiTaskDense(self.specific_features[-1], self.n_tasks)(x)
        return x.squeeze().T


class SirenMLP(nn.Module):
    """Sine-activated neural network, aka SIREN. Be sure to 
    normalize inputs between -1 and 1!"""

    features: Sequence[int]
    omega_0: int = 30

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for layer_idx, feature in enumerate(self.features[:-1]):
            x = SineLayer(feature, omega=self.omega_0, is_first=layer_idx == 0)
        return nn.Dense(self.features[-1])(x)
