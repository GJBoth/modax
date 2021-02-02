import dataclasses
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Convergence:
    patience: int = 200
    delta: float = 1e-3
    start_iteration = None
    start_norm = None

    def __call__(self, iteration, coeffs):
        coeff_norm = jnp.linalg.norm(coeffs)

        # Initialize if doesn't exist
        if self.start_norm is None:
            self.start_norm = coeff_norm
            self.start_iteration = iteration
            converged = False

        # Check if change is smaller than delta and if we've exceeded patience
        elif jnp.abs(self.start_norm - coeff_norm).item() < self.delta:
            if (iteration - self.start_iteration) >= self.patience:
                converged = True
            else:
                converged = False

        # If not, reset and keep going
        else:
            self.start_norm = coeff_norm
            self.start_iteration = iteration
            converged = False

        return converged
