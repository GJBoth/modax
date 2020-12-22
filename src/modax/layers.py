from typing import Callable
from jax import lax
from flax import linen as nn
import jax.numpy as jnp
import jax
from modax.losses import mse


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
