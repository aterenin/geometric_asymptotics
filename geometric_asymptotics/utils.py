import jax.numpy as jnp
from jaxtyping import Float, Array
from tqdm.auto import tqdm

def batch_predict_mean(posterior, test_data: Float[Array, "N D"]):
    num_batches = 1 if test_data.shape[0] < 5000 else 25
    test_data_batched = jnp.array_split(test_data, num_batches)
    output = [posterior(x) for x in tqdm(test_data_batched)]

    return jnp.concatenate([p.mean() for p in output])

def batch_predict_variance(posterior, test_data: Float[Array, "N D"]):
    num_batches = 1 if test_data.shape[0] < 5000 else 25
    test_data_batched = jnp.array_split(test_data, num_batches)
    output = [posterior(x) for x in tqdm(test_data_batched)]

    return jnp.concatenate([p.variance() for p in output])