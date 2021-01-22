from jax import jit, numpy as jnp

from modax.training.convergence import Convergence
from .sparsity_scheduler import mask_scheduler
from .logging import Logger
from flax.core import freeze
from ..utils import create_stateful_update
from sklearn.model_selection import train_test_split
from .convergence import Convergence


def train(
    model,
    optimizer,
    state,
    loss_fn,
    mask_update_fn,
    X,
    y,
    max_epochs=1e4,
    split=0.2,
    rand_seed=42,
):
    # Making test / train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=rand_seed
    )

    # Creating update functions
    update = create_stateful_update(loss_fn, model=model, x=X_train, y=y_train)
    validation_metric = jit(
        lambda opt, state: loss_fn(opt.target, state, model, X_test, y_test)[1][1]
    )
    logger = Logger()
    scheduler = mask_scheduler()
    converged = Convergence()

    for epoch in jnp.arange(max_epochs):
        (optimizer, state), train_metrics, output = update(optimizer, state)
        prediction, dt, theta, coeffs = output

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {train_metrics['loss']}")

        if epoch % 25 == 0:
            val_metrics = validation_metric(optimizer, state)
            metrics = {
                **train_metrics,
                "val_mse": val_metrics["mse"],
                "val_reg": val_metrics["reg"],
            }
            logger.write(metrics, epoch)

            apply_sparsity, optimizer = scheduler(val_metrics["mse"], epoch, optimizer)

            if apply_sparsity:
                mask = mask_update_fn(theta, dt)
                state = freeze({"vars": {"MaskedLeastSquares_0": {"mask": mask}}})

        if converged(epoch, coeffs):
            mask = mask_update_fn(theta, dt)
            print(f"Converged at epoch {epoch} with mask {mask[:, None]}.")
            break

    logger.close()
    return optimizer, state


def train_probabilistic(
    model,
    optimizer,
    state,
    loss_fn,
    mask_update_fn,
    X,
    y,
    max_epochs=1e4,
    split=0.2,
    rand_seed=42,
    **loss_fn_kwargs,
):
    # Making test / train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=rand_seed
    )

    # Creating update functions
    update = create_stateful_update(
        loss_fn, model=model, x=X_train, y=y_train, **loss_fn_kwargs
    )
    validation_metric = jit(
        lambda opt, state: loss_fn(
            opt.target, state, model, X_test, y_test, **loss_fn_kwargs
        )[1][1]
    )
    logger = Logger()
    scheduler = mask_scheduler()
    converged = Convergence()

    for epoch in jnp.arange(max_epochs):
        (optimizer, state), train_metrics, output = update(optimizer, state)
        prediction, dt, theta, coeffs = output

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {train_metrics['loss']}")

        if epoch % 25 == 0:
            val_metrics = validation_metric(
                optimizer, state
            )  # this is not correct, val needs to calculated with forward val.
            metrics = {
                **train_metrics,
                "val_p_mse": val_metrics["p_mse"],
                "val_p_reg": val_metrics["p_reg"],
                "val_mse": val_metrics["mse"],
                "val_reg": val_metrics["reg"],
            }
            logger.write(metrics, epoch)

            apply_sparsity, optimizer = scheduler(
                val_metrics["mse"], epoch, optimizer
            )  # we need to find the minimum neg LL of the mse

            if apply_sparsity:
                mask = mask_update_fn(theta, dt)
                state = freeze({"vars": {"MaskedLeastSquares_0": {"mask": mask}}})

        if converged(epoch, coeffs):
            mask = mask_update_fn(theta, dt)
            print(f"Converged at epoch {epoch} with mask {mask[:, None]}.")
            break

    logger.close()
    return optimizer, state


def train_probabilistic_mse(
    model, optimizer, state, loss_fn, X, y, max_epochs=1e4, split=0.2, rand_seed=42,
):
    # Making test / train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=rand_seed
    )

    # Creating update functions
    update = create_stateful_update(loss_fn, model=model, x=X_train, y=y_train)
    validation_metric = jit(
        lambda opt, state: loss_fn(opt.target, state, model, X_test, y_test)[1][1]
    )
    logger = Logger()
    scheduler = mask_scheduler()

    for epoch in jnp.arange(max_epochs):
        (optimizer, state), train_metrics, output = update(optimizer, state)
        prediction, dt, theta, coeffs = output

        if epoch % 1000 == 0:
            print(f"Loss step {epoch}: {train_metrics['loss']}")

        if epoch % 25 == 0:
            val_metrics = validation_metric(
                optimizer, state
            )  # this is not correct, val needs to calculated with forward val.
            metrics = {
                **train_metrics,
                "val_p_mse": val_metrics["p_mse"],
                "val_p_reg": val_metrics["p_reg"],
                "val_mse": val_metrics["mse"],
                "val_reg": val_metrics["reg"],
            }
            logger.write(metrics, epoch)

            apply_sparsity, optimizer = scheduler(
                val_metrics["mse"], epoch, optimizer
            )  # we need to find the minimum neg LL of the mse

            if apply_sparsity:
                break

    logger.close()
    return optimizer, state
