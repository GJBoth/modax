from jax import jit, numpy as jnp

from .schedulers.convergence import Convergence
from .schedulers.sparsity_scheduler import mask_scheduler
from .logging import Logger
from flax.core import freeze


def train_max_iter(update_fn, optimizer, state, max_epochs):
    """Run update_fn for max_epochs iteration.
    """
    logger = Logger()
    for epoch in jnp.arange(max_epochs):
        (optimizer, state), metrics, output = update_fn(optimizer, state)

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {metrics['loss']}")

        if epoch % 25 == 0:
            logger.write(metrics, epoch)
    logger.close()
    return optimizer, state


def train_early_stop(
    update_fn, validation_fn, optimizer, state, max_epochs=1e4, **early_stop_args
):
    """Run update_fn until given validation metric validation_fn increases.
    """
    logger = Logger()
    check_early_stop = mask_scheduler(**early_stop_args)
    for epoch in jnp.arange(max_epochs):
        (optimizer, state), metrics, output = update_fn(optimizer, state)

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {metrics['loss']}")

        if epoch % 25 == 0:
            val_metric = validation_fn(optimizer, state)
            stop_training, optimizer = check_early_stop(val_metric, epoch, optimizer)
            metrics = {**metrics, "validation_metric": val_metric}
            logger.write(metrics, epoch)
            if stop_training:
                print("Converged.")
                break
    logger.close()
    return optimizer, state


def train_full(
    update_fn,
    validation_fn,
    mask_update_fn,
    optimizer,
    state,
    max_epochs=1e4,
    convergence_args={"patience": 200, "delta": 1e-3},
    mask_update_args={"patience": 500, "delta": 1e-5, "periodicity": 200},
):

    logger = Logger()
    converged = Convergence(**convergence_args)
    update_mask = mask_scheduler(**mask_update_args)

    for epoch in jnp.arange(max_epochs):
        (optimizer, state), metrics, output = update_fn(optimizer, state)
        prediction, dt, theta, coeffs = output

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {metrics['loss']}")

        if epoch % 25 == 0:
            val_metric = validation_fn(optimizer, state)
            metrics = {**metrics, "validation_metric": val_metric}
            logger.write(metrics, epoch)
            apply_sparsity, optimizer = update_mask(val_metric, epoch, optimizer)

            if apply_sparsity:
                mask = mask_update_fn(theta, dt)
                state = freeze({"vars": {"LeastSquares_0": {"mask": mask}}})

        if converged(epoch, coeffs):
            mask = mask_update_fn(theta, dt)
            print(f"Converged at epoch {epoch} with mask {mask[:, None]}.")
            break

    logger.close()
    return optimizer, state

