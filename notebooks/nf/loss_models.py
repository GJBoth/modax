import jax.numpy as jnp
from modax.models import DeepmodBase
from modax.layers.feature_generators import library_backward
from modax.layers.regression import LeastSquares
from typing import Sequence, Tuple
from nf import AmortizedNormalizingFlow


def loss_fn(params, state, model, x):
    log_pz, log_det = model.apply(params, x)
    log_p = log_pz + log_det
    loss = -jnp.sum(log_pz + log_det)
    metrics = {"loss": loss}
    return loss, (state, metrics, (log_p, log_pz, log_det))


def DeepmodNF(
    network_shape: Sequence[int], n_flow_layers: int, library_orders: Tuple[int, int]
):
    return DeepmodBase(
        AmortizedNormalizingFlow,
        (network_shape, n_flow_layers,),
        library_backward,
        (*library_orders,),
        LeastSquares,
        (),
    )

