import jax.numpy as jnp
import jax


def ridge(X, y, l):
    """Ridge regression using augmente data. X can have dimensions."""
    l_normed = jnp.diag(jnp.sqrt(l) * jnp.linalg.norm(X, axis=0))
    l_normed = jax.ops.index_update(
        l_normed, jax.ops.index[0, 0], 0.0
    )  # shouldnt apply l2 to offset
    X_augmented = jnp.concatenate([X, l_normed], axis=0)
    y_augmented = jnp.concatenate([y, jnp.zeros((X.shape[1], 1))], axis=0)

    coeffs = jnp.linalg.lstsq(X_augmented, y_augmented)[0]
    return coeffs
