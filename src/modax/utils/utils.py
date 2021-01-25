from jax import custom_vjp


@custom_vjp
def no_grad(x):
    return x


def no_grad_fwd(x):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return no_grad(x), ()


def no_grad_bwd(res, g):
    return None


no_grad.defvjp(no_grad_fwd, no_grad_bwd)
