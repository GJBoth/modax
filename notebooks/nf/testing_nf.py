import jax.numpy as jnp
import jax.random as random
from flax import optim
from modax.training import create_update, train_max_iter
from nf import NormalizingFlow

key = random.PRNGKey(42)
# Making sample dataset
n_samples = 100
n_dims = 1
x_samples = 0.5 * random.normal(key, (n_samples, n_dims))


def loss_fn(params, state, model, x):
    log_p = model.apply(params, x)
    loss = -jnp.sum(log_p)
    metrics = {"loss": loss}
    return loss, (state, metrics, log_p)


# Initializing model
model = NormalizingFlow(10)
params = model.init(key, x_samples)
optimizer_def = optim.Adam(learning_rate=2e-3, beta1=0.99, beta2=0.99)
optimizer = optimizer_def.create(params)

# Compiling train step
update = create_update(loss_fn, (model, x_samples))
train_max_iter(update, optimizer, None, 5000)
