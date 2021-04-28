# %%
import jax.numpy as jnp
import jax.random as random
from flax import optim
from modax.training import create_update, train_max_iter
from nf import NormalizingFlow
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import unfreeze, freeze

key = random.PRNGKey(42)
# %% Making sample dataset
n_samples = 100
n_dims = 1
x_samples = 0.5 * random.normal(key, (n_samples, n_dims))

# %% Initializing model
model = NormalizingFlow(10)
params = model.init(key, x_samples)


# %%

# %%
