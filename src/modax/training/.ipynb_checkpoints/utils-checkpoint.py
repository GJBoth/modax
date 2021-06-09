from jax import jit, value_and_grad
from functools import partial
import jax.profiler

def create_update(loss_fn, loss_fn_args):
    def step(opt, state, loss_fn, loss_fn_args):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (updated_state, metrics, output)), grad = grad_fn(
            opt.target, state, *loss_fn_args
        )
        opt = opt.apply_gradient(grad)
        jax.profiler.save_device_memory_profile(f"memory.prof")
        return (opt, updated_state), metrics, output

    return jit(partial(step, loss_fn=loss_fn, loss_fn_args=loss_fn_args))

