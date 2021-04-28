from typing import Callable
from jax import lax
from flax import linen as nn
import jax.numpy as jnp
from .initializers import siren_kernel_init

class MultiTaskDense(nn.Module):
    features: int
    n_tasks: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (self.n_tasks, inputs.shape[-1], self.features)
        )
        y = lax.dot_general(
            inputs, kernel, dimension_numbers=(((2,), (1,)), ((0,), (0,)))
        )
        bias = self.param("bias", self.bias_init, (self.n_tasks, 1, self.features))
        y = y + bias
        return y


class SineLayer(nn.Module):
    """Basic sine layer with scaling for siren."""
    features: int
    omega: float
    is_first: bool

    def __setup__(self):
        self.linear = nn.Dense(features=self.features, 
        kernel_init=siren_kernel_init(self.omega, self.is_first))

    def __call__(self, inputs):
        return jnp.sin(self.omega * self.linear(inputs))

