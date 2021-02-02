# %% Imports
import jax
from jax import jit, value_and_grad, numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple
from modax.feature_generators import library_backward
from modax.networks import MLP
from modax.losses import neg_LL

from functools import partial
from jax import lax


# Forward solver
@partial(jit, static_argnums=(0,))
def fwd_solver(f, z_init, tol=1e-4):
    def cond_fun(carry):
        z_prev, z = carry
        return (
            jnp.linalg.norm(z_prev[:-1] - z[:-1]) > tol
        )  # for numerical reasons, we check the change in alpha

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


# Update functions and evidence for bayesian ridge
@jit
def bayes_ridge_update(prior_params, y, X, hyper_prior_params):
    # Unpacking parameters
    alpha_prev, beta_prev = prior_params
    a, b = hyper_prior_params

    # Preparing some matrices
    X_normed = X / jnp.linalg.norm(X, axis=0)
    gram = X_normed.T @ X_normed
    eigvals = jnp.linalg.eigvalsh(gram)

    # Calculating intermediate matrices
    n_samples, n_terms = X.shape
    gamma_ = jnp.sum((beta_prev * eigvals) / (alpha_prev + beta_prev * eigvals))
    S = jnp.linalg.inv(beta_prev * gram + alpha_prev * jnp.eye(n_terms))
    mn = beta_prev * S @ X_normed.T @ y

    # Update estimate
    alpha = gamma_ / jnp.sum(mn ** 2)
    beta = (n_samples - gamma_ + 2 * a) / (jnp.sum((y - X_normed @ mn) ** 2) + 2 * b)

    return jnp.stack([alpha, beta], axis=0)


@jit
def bayes_ridge_update_efficient(prior_params, y, X, gram, eigvals, hyper_prior_params):
    # Unpacking parameters
    alpha_prev, beta_prev = prior_params
    a, b = hyper_prior_params

    # Calculating intermediate matrices
    n_samples, n_terms = X.shape
    gamma_ = jnp.sum((beta_prev * eigvals) / (alpha_prev + beta_prev * eigvals))
    S = jnp.linalg.inv(beta_prev * gram + alpha_prev * jnp.eye(n_terms))
    mn = beta_prev * S @ X.T @ y

    # Update estimate
    alpha = gamma_ / jnp.sum(mn ** 2)
    beta = (n_samples - gamma_ + 2 * a) / (jnp.sum((y - X @ mn) ** 2) + 2 * b)

    return jnp.stack([alpha, beta], axis=0)


@jit
def evidence(prior_params, y, X, hyper_prior_params):
    alpha, beta = prior_params
    a, b = hyper_prior_params

    n_samples, n_terms = X.shape
    A = alpha * jnp.eye(n_terms) + beta * X.T @ X
    mn = beta * jnp.linalg.inv(A) @ X.T @ y

    E = beta * jnp.sum((y - X @ mn) ** 2) + alpha * jnp.sum(mn ** 2)
    loss = 0.5 * (
        n_terms * jnp.log(alpha)
        + n_samples * jnp.log(beta)
        - E
        - jnp.linalg.slogdet(A)[1]
        - n_samples * jnp.log(2 * jnp.pi)
    )

    # following tipping, numerically more stable if a, b -> 0 but doesn't have constant terms.
    loss += a * jnp.log(beta) - b * beta
    return loss, mn


# Custom backprop function for iterative methods
@partial(jax.custom_vjp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def fixed_point_solver(f, args, z_init, tol=1e-5):
    z_star = fwd_solver(lambda z: f(z, *args), z_init=z_init, tol=tol)
    return z_star


@partial(jit, static_argnums=(0,))
def fixed_point_solver_fwd(f, args, z_init, tol):
    z_star = fixed_point_solver(f, args, z_init, tol)
    return z_star, (z_star, tol, args)


@partial(jit, static_argnums=(0,))
def fixed_point_solver_bwd(f, res, z_star_bar):
    z_star, tol, args = res
    _, vjp_a = jax.vjp(lambda args: f(z_star, *args), args)
    _, vjp_z = jax.vjp(lambda z: f(z, *args), z_star)
    res = vjp_a(
        fwd_solver(
            lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(z_star), tol=tol
        )
    )
    return (*res, None, None)  # None for init and tol


fixed_point_solver.defvjp(fixed_point_solver_fwd, fixed_point_solver_bwd)

# Flax layers for bayesian regression
class BayesianRegression(nn.Module):
    hyper_prior: Tuple
    tol: float = 1e-3

    def setup(self):
        self.update = lambda prior, y, X: bayes_ridge_update(
            prior, y, X, self.hyper_prior
        )

    @nn.compact
    def __call__(self, inputs):
        is_initialized = self.has_variable("bayes", "z")
        z_init = self.variable(
            "bayes", "z", lambda y: jnp.stack([1, 1 / jnp.var(y)], axis=0), inputs[0]
        )

        z_star = fixed_point_solver(self.update, inputs, z_init.value, tol=self.tol)
        if is_initialized:
            z_init.value = z_star
        return z_star


class BayesianRegressionEfficient(nn.Module):
    hyper_prior: Tuple
    tol: float = 1e-3

    def setup(self):
        self.update = lambda prior, y, X, gram, eigvals: bayes_ridge_update_efficient(
            prior, y, X, gram, eigvals, self.hyper_prior
        )

    @nn.compact
    def __call__(self, inputs):
        is_initialized = self.has_variable("bayes", "z")
        z_init = self.variable(
            "bayes", "z", lambda y: jnp.stack([1, 1 / jnp.var(y)], axis=0), inputs[0]
        )

        # These things remain constant every iteration step, so we precalculate them
        y, X = inputs
        X_normed = X / jnp.linalg.norm(X, axis=0, keepdims=True)
        gram = X_normed.T @ X_normed
        eigvals = jnp.linalg.eigvalsh(gram)

        z_star = fixed_point_solver(
            self.update, (y, X_normed, gram, eigvals), z_init.value, tol=self.tol
        )
        if is_initialized:
            z_init.value = z_star
        return z_star


# Deepmodel model for bayesian regression
class Deepmod(nn.Module):
    features: Sequence[int]
    hyper_prior: Tuple
    tol: float = 1e-3

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        z = BayesianRegression(self.hyper_prior, self.tol)((dt, theta))
        return prediction, dt, theta, z


# Loss function to use with bayes regression
def loss_fn_pinn_bayes_regression(params, state, model, x, y):
    """first argument should always be params!"""
    variables = {"params": params, **state}
    (prediction, dt, theta, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    # MSE
    sigma_ml = jnp.mean((prediction - y) ** 2)
    tau = 1 / sigma_ml
    MSE = neg_LL(prediction, y, tau)

    # Reg
    theta_normed = theta / jnp.linalg.norm(theta, axis=0, keepdims=True)
    Reg, mn = evidence(z, dt, theta_normed, model.hyper_prior)
    loss = MSE - Reg
    metrics = {
        "loss": loss,
        "mse": MSE,
        "reg": Reg,
        "coeff": mn,
        "tau": tau,
        "beta": z[1],
        "alpha": z[0],
    }
    return loss, (updated_state, metrics)


# Function to create fast update for flax models with variables
def create_update(loss_fn, *args, **kwargs):
    def step(opt, state, loss_fn, *args, **kwargs):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (updated_state, metrics)), grad = grad_fn(
            opt.target, state, *args, **kwargs
        )
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.

        return (opt, updated_state), metrics

    return jit(lambda opt, state: step(opt, state, loss_fn, *args, **kwargs))


class DeepmodSBL(nn.Module):
    features: Sequence[int]
    hyper_prior: Tuple
    tol: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        prediction, dt, theta = library_backward(MLP(self.features), inputs)
        z = SparseBayesianLearning(self.hyper_prior, self.tol)((dt, theta))
        return prediction, dt, theta, z


class SparseBayesianLearning(nn.Module):
    hyper_prior: Tuple
    tol: float = 1e-4

    def setup(self):
        self.update = lambda prior, y, X: SBL_update(prior, y, X, self.hyper_prior)

    @nn.compact
    def __call__(self, inputs):
        is_initialized = self.has_variable("bayes", "z")
        z_init = self.variable(
            "bayes",
            "z",
            lambda X: jnp.concatenate(
                [jnp.ones((X[1].shape[1],)), jnp.ones((1,)) / jnp.var(X[0])], axis=0
            ),
            inputs,
        )

        z_star = fixed_point_solver(self.update, inputs, z_init.value, tol=self.tol)

        if is_initialized:
            z_init.value = z_star
        return z_star


def loss_fn_SBL(params, state, model, x, y):
    """ first argument should always be params!
    """
    variables = {"params": params, **state}
    (prediction, dt, theta, z), updated_state = model.apply(
        variables, x, mutable=list(state.keys())
    )

    # MSE
    sigma_ml = jnp.mean((prediction - y) ** 2)
    tau = 1 / sigma_ml
    MSE = neg_LL(prediction, y, tau)

    # Reg
    Reg, mn = evidence_SBL(z, dt, theta, model.hyper_prior)
    loss = MSE - Reg
    metrics = {
        "loss": loss,
        "mse": MSE,
        "reg": Reg,
        "coeff": mn,
        "tau": tau,
        "beta": z[-1],
        "alpha": z[:-1],
    }
    return loss, (updated_state, metrics)


@jit
def evidence_SBL(prior_params, y, X, hyper_prior_params):
    alpha, beta = prior_params[:-1], prior_params[-1]
    a, b = hyper_prior_params

    n_samples, n_terms = X.shape
    A = jnp.diag(alpha) + beta * X.T @ X
    mn = beta * jnp.linalg.inv(A) @ X.T @ y

    E = beta * jnp.sum((y - X @ mn) ** 2) + (mn.T @ jnp.diag(alpha) @ mn).squeeze()
    loss = 0.5 * (
        jnp.sum(jnp.log(alpha))
        + n_samples * jnp.log(beta)
        - E
        - jnp.linalg.slogdet(A)[1]
    )

    # following tipping, numerically more stable if a, b -> 0 but doesn't have constant terms.
    loss += a * jnp.log(beta) - b * beta
    return loss, mn


def SBL_update(prior_params, y, X, hyper_prior_params):
    # Unpacking parameters
    alpha_prev, beta_prev = prior_params[:-1], prior_params[-1]
    a, b = hyper_prior_params

    # Calculating intermediate matrices
    n_samples, n_terms = X.shape
    Sigma = jnp.linalg.inv(beta_prev * X.T @ X + jnp.diag(alpha_prev))
    mu = beta_prev * Sigma @ X.T @ y
    gamma = 1 - alpha_prev * jnp.diag(Sigma)

    # Updating
    cap = 1e6
    alpha = jnp.minimum(gamma / (mu ** 2).squeeze(), cap)
    beta = (n_samples - jnp.sum(gamma) + 2 * a) / (jnp.sum((y - X @ mu) ** 2) + 2 * b)

    return jnp.concatenate([alpha, beta[None]], axis=0)
