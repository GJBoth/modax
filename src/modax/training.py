from jax import value_and_grad, jit


def create_update(loss_fn, *args, **kwargs):
    """Constructs a fast update given a loss function.
    """

    def step(opt, loss_fn, *args, **kwargs):
        grad_fn = value_and_grad(loss_fn, argnums=0)
        loss, grad = grad_fn(opt.target, *args, **kwargs)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return opt, loss

    return jit(lambda opt: step(opt, loss_fn, *args, **kwargs))
