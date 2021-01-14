import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LassoCV


class ThresholdedLasso:
    def __init__(self, threshold=0.1, *args, **kwargs):
        self.threshold = threshold
        self.reg = LassoCV(fit_intercept=False, *args, **kwargs)

    def fit(self, X, y):
        return self.reg.fit(np.array(X), np.array(y.squeeze())).coef_

    def __call__(self, X, y, threshold=0.1):
        X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)
        y_normed = y / jnp.linalg.norm(y, axis=0, keepdims=True)
        coeffs = self.fit(X_normed, y_normed)
        mask = np.abs(coeffs) > threshold
        return mask
