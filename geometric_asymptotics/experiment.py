from typing import Tuple
import numpy as np
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxlinop import identity
from jaxtyping import Float, Array
from jaxutils import Dataset
import optax as ox
from tqdm.auto import tqdm
from pathlib import Path
import jaxkern
from jaxkern.base import AbstractKernel
import gpjax as gpx
from gpjax.parameters import ParameterState
from geometric_kernels.frontends.jax.gpjax import GPJaxGeometricKernel
from geometric_kernels.spaces import DiscreteSpectrumSpace
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from .kernel import ScaleKernel
from .io import load_space

def draw_random_data_from_prior(kernel, params, key, num_data, num_nodes, noise_variance):
    (k1,k2) = jr.split(key)
    X = jr.randint(k1, shape=(num_data, 1), minval=0, maxval=num_nodes)
    K = kernel.gram(params, X)
    y = (K + identity(num_data) * noise_variance).to_root() @ jr.normal(k2, (num_data, 1))
    return X, y
    
def generate_kernel_and_training_data(
    space: DiscreteSpectrumSpace, 
    coordinates: Float[Array, "N D"], 
    config: dict,
    key: jr.PRNGKey
    ):
    alpha = config["alpha"]
    kappa = config["kappa"]
    nu = config["nu"]
    dimension_factor = config["dimension_factor"]
    noise_variance = config["noise_variance"]
    truncation_level = config["truncation_level"]
    num_data = config["num_data"]

    base_kernel = MaternKarhunenLoeveKernel(space, truncation_level)
    kernel = ScaleKernel(GPJaxGeometricKernel(base_kernel))
    kernel_extr = jaxkern.Matern52()

    num_nodes = coordinates.shape[0]

    X_test = jnp.expand_dims(jnp.arange(num_nodes),-1)
    X_test_extr = coordinates[X_test.squeeze(),:]

    (k1,key) = jr.split(key)
    prior_kernel_params = kernel.init_params(k1)
    prior_kernel_params['variance'] = jnp.array(alpha)
    prior_kernel_params['lengthscale'] = jnp.array(kappa)
    prior_kernel_params['nu'] = jnp.array(nu + dimension_factor)
    normalization_factor = jnp.mean(jnp.vectorize(lambda x: kernel.gram(prior_kernel_params,x[None,...]).to_dense().squeeze(), signature="(d)->()")(X_test))
    prior_kernel_params['variance'] = jnp.array(alpha / normalization_factor)

    (k2, key) = jr.split(key)
    X, y = draw_random_data_from_prior(kernel, prior_kernel_params, k2, num_data, num_nodes, noise_variance)
    X_extr = coordinates[X.squeeze(),:]

    return (kernel, kernel_extr, prior_kernel_params, normalization_factor), (X, X_extr, y), (X_test, X_test_extr)



def compute_predictions(
    kernels: Tuple[AbstractKernel,AbstractKernel],
    training_data: Tuple[Float[Array, "N 1"], Float[Array, "N D"], Float[Array, "N 1"]],
    test_data: Tuple[Float[Array, "N 1"], Float[Array, "N D"]],
    config: dict,
    key: jr.PRNGKey
    ):
    (kernel, _, _, normalization_factor) = kernels
    (X, X_extr, y) = training_data
    (X_test, X_test_extr) = test_data

    subset_size_int = config["subset_size_int"]
    num_data = config["num_data"]
    num_plot = config["num_plot"]
    alpha = config["alpha"]
    kappa = config["kappa"]
    nu = config["nu"]
    dimension_factor = config["dimension_factor"]
    noise_variance = config["noise_variance"]

    data_int = Dataset(X=X[:subset_size_int,:], y=y[:subset_size_int,:].reshape(-1, 1))
    prior_int = gpx.Prior(kernel=kernel)
    likelihood_int = gpx.likelihoods.Gaussian(num_datapoints=num_data)
    posterior_int = likelihood_int * prior_int

    (k,key) = jr.split(key)
    params_int, _, _ = gpx.initialise(posterior_int, k).unpack()
    params_int['kernel']['variance'] = jnp.array(alpha / normalization_factor)
    params_int['kernel']['lengthscale'] = jnp.array(kappa)
    params_int['kernel']['nu'] = jnp.array(nu + dimension_factor)
    params_int['likelihood']['obs_noise'] = jnp.array(noise_variance)

    pred_mean = (X_extr[:subset_size_int,:], posterior_int.predict(params_int, data_int)(X[:subset_size_int,:]).mean())
    test_pred_mean = (X_test_extr[:num_plot,:], posterior_int.predict(params_int, data_int)(X_test[:num_plot,:]).mean())
    test_pred_var = (X_test_extr[:num_plot,:], posterior_int.predict(params_int, data_int)(X_test[:num_plot,:]).variance())
    
    return (pred_mean, test_pred_mean, test_pred_var, (data_int, posterior_int, params_int))



def compute_extrinsic_length_scale(
    kernels: Tuple[AbstractKernel,AbstractKernel],
    training_data: Tuple[Float[Array, "N 1"], Float[Array, "N D"], Float[Array, "N 1"]],
    config: dict,
    key: jr.PRNGKey,
    ):
    (_, kernel_extr, _, _) = kernels
    (_, X_extr, y) = training_data

    subset_size_ls = config["subset_size_ls"]
    num_data = config["num_data"]
    alpha = config["alpha"]
    kappa_ls_initial = config["kappa_ls_initial"]
    noise_variance = config["noise_variance"]
    num_ls_iters = config["num_ls_iters"]

    data_ls = Dataset(X=X_extr[:subset_size_ls,:], y=y[:subset_size_ls,:].reshape(-1,1))
    prior_ls = gpx.Prior(kernel=kernel_extr)
    likelihood_ls = gpx.likelihoods.Gaussian(num_datapoints=num_data)
    posterior_ls = likelihood_ls * prior_ls

    (k3,key) = jr.split(key)
    params_ls, trainables_ls, bijectors_ls = gpx.initialise(posterior_ls, k3).unpack()
    params_ls['kernel']['variance'] = jnp.array(alpha)
    params_ls['kernel']['lengthscale'] = jnp.array(kappa_ls_initial)
    params_ls['likelihood']['obs_noise'] = jnp.array(noise_variance)
    trainables_ls['kernel']['variance'] = False
    trainables_ls['likelihood']['obs_noise'] = False

    parameter_state_ls = ParameterState(params_ls, trainables_ls, bijectors_ls)
    negative_mll_ls = jit(posterior_ls.marginal_log_likelihood(data_ls, negative = True))
    optimiser_ls = ox.adam(learning_rate=5e-3)

    inference_state_ls = gpx.fit(
        objective=negative_mll_ls,
        parameter_state=parameter_state_ls,
        optax_optim=optimiser_ls,
        num_iters=num_ls_iters,
    )

    learned_params_ls, _ = inference_state_ls.unpack()
    kappa_extr = float(learned_params_ls['kernel']['lengthscale'])
    return (kappa_extr, posterior_ls, learned_params_ls)




def compute_non_asymptotic_expected_error(
    kernels: Tuple[AbstractKernel,AbstractKernel],
    training_data: Tuple[Float[Array, "N 1"], Float[Array, "N D"], Float[Array, "N 1"]],
    test_data: Tuple[Float[Array, "N 1"], Float[Array, "N D"]],
    num_nodes: int,
    kappa_extr: float,
    config: dict,
    key: jr.PRNGKey,
    ):
    (kernel, kernel_extr, _, normalization_factor) = kernels
    (X, X_extr, y) = training_data
    (X_test, X_test_extr) = test_data

    subset_size = config["subset_size"]
    num_data_grid_points = config["num_data_grid_points"]
    alpha = config["alpha"]
    kappa = config["kappa"]
    nu = config["nu"]
    dimension_factor = config["dimension_factor"]
    noise_variance = config["noise_variance"]

    expected_error_idxs = np.linspace(0,subset_size,num_data_grid_points,dtype=int)
    expected_errors = np.zeros_like(expected_error_idxs, dtype=float)
    expected_errors_extr = np.zeros_like(expected_error_idxs, dtype=float)
    expected_errors_approx_intr = np.zeros_like(expected_error_idxs, dtype=float)
    expected_errors_std = np.zeros_like(expected_error_idxs, dtype=float)
    expected_errors_extr_std = np.zeros_like(expected_error_idxs, dtype=float)
    expected_errors_approx_intr_std = np.zeros_like(expected_error_idxs, dtype=float)

    (k4,key) = jr.split(key)
    subset_idxs = jr.choice(k4, jnp.arange(num_nodes), shape = (subset_size,), replace=False)
    X_subset = X_test[subset_idxs,:]
    X_subset_extr = X_test_extr[subset_idxs,:]

    for (idx,num_subset) in enumerate(tqdm(expected_error_idxs)):
        data = Dataset(X=X[:num_subset,:], y=y[:num_subset,:].reshape(-1, 1))
        prior = gpx.Prior(kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_subset)
        posterior = likelihood * prior

        (k1,key) = jr.split(key)
        params, _, _ = gpx.initialise(posterior, k1).unpack()
        params['kernel']['variance'] = jnp.array(alpha / normalization_factor)
        params['kernel']['lengthscale'] = jnp.array(kappa)
        params['kernel']['nu'] = jnp.array(nu + dimension_factor)
        params['likelihood']['obs_noise'] = jnp.array(noise_variance)

        test_variance = posterior(params, data)(X_subset).variance()
        expected_errors[idx] = jnp.mean(test_variance)
        expected_errors_std[idx] = jnp.std(test_variance)

        data_extr = Dataset(X=X_extr[:num_subset,:], y=y[:num_subset,:].reshape(-1,1))
        prior_extr = gpx.Prior(kernel=kernel_extr)
        likelihood_extr = gpx.likelihoods.Gaussian(num_datapoints=num_subset)
        posterior_extr = likelihood_extr * prior_extr

        (k2,key) = jr.split(key)
        params_extr, _, _ = gpx.initialise(posterior_extr, k2).unpack()
        params_extr['kernel']['variance'] = jnp.array(alpha)
        params_extr['kernel']['lengthscale'] = jnp.array(kappa_extr)
        params_extr['likelihood']['obs_noise'] = jnp.array(noise_variance)

        jitter = 1e-10 if params_extr['likelihood']['obs_noise'].dtype == 'float64' else 1e-5
        
        vector_term = (kernel_extr.gram(params_extr['kernel'], data_extr.X) + identity(data_extr.X.shape[0]) * noise_variance).solve(
            kernel_extr.cross_covariance(params_extr['kernel'], data_extr.X, X_subset_extr)
        )
        matrix_term = kernel.gram(params['kernel'], X_subset).to_dense() - (
            kernel.cross_covariance(params['kernel'], X_subset, data.X) @ vector_term
        )

        first_term = jnp.diag(matrix_term.T @
            (kernel.gram(params['kernel'], X_subset) + identity(X_subset.shape[0]) * jitter).solve(
                matrix_term
            )
        )
        second_term = noise_variance * vector_term.T @ vector_term
        
        vector_term_intr = (kernel.gram(params['kernel'], data.X) + identity(data.X.shape[0]) * noise_variance).solve(
            kernel.cross_covariance(params['kernel'], data.X, X_subset)
        )
        matrix_term_intr = kernel.gram(params['kernel'], X_subset).to_dense() - (
            kernel.cross_covariance(params['kernel'], X_subset, data.X) @ vector_term_intr
        )

        first_term_intr = jnp.diag(matrix_term_intr.T @
            (kernel.gram(params['kernel'], X_subset) + identity(X_subset.shape[0]) * jitter).solve(
                matrix_term_intr
            )
        )
        second_term_intr = noise_variance * vector_term_intr.T @ vector_term_intr

        expected_errors_extr[idx] = jnp.mean(first_term + second_term)
        expected_errors_approx_intr[idx] = jnp.mean(first_term_intr + second_term_intr)
        expected_errors_extr_std[idx] = jnp.std(first_term + second_term)
        expected_errors_approx_intr_std[idx] = jnp.std(first_term_intr + second_term_intr)

    output = {
        "expected_error_idxs": expected_error_idxs,
        "expected_errors": expected_errors,
        "expected_errors_extr": expected_errors_extr,
        "expected_errors_approx_intr": expected_errors_approx_intr,
        "expected_errors_std": expected_errors_std,
        "expected_errors_extr_std": expected_errors_extr_std,
        "expected_errors_approx_intr_std": expected_errors_approx_intr_std,
    }
    
    return output



def run_experiment(
    seed: int,
    config: dict,
    ):
    (space, coordinates) = load_space(config["source"])

    key = jr.PRNGKey(seed)
    (k1,key) = jr.split(key)

    (kernels, training_data, test_data) = generate_kernel_and_training_data(space, coordinates, config, k1)
    
    (k2,key) = jr.split(key)
    (kappa_extr, _, _) = compute_extrinsic_length_scale(kernels, training_data, config, k2)

    (k3,key) = jr.split(key)
    num_nodes = coordinates.shape[0]
    output = compute_non_asymptotic_expected_error(kernels, training_data, test_data, num_nodes, kappa_extr, config, k3)

    manifold = config["name"]
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_error_idxs_{seed}.csv", output["expected_error_idxs"], delimiter=',', fmt='%i')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_{seed}.csv", output["expected_errors"], delimiter=',')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_extr_{seed}.csv", output["expected_errors_extr"], delimiter=',')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_approx_intr_{seed}.csv", output["expected_errors_approx_intr"], delimiter=',')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_error_idxs_std_{seed}.csv", output["expected_error_idxs"][1:], delimiter=',', fmt='%i')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_std_{seed}.csv", output["expected_errors_std"][1:], delimiter=',')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_extr_std_{seed}.csv", output["expected_errors_extr_std"][1:], delimiter=',')
    np.savetxt(Path.cwd() / "results" / f"{manifold}_expected_errors_approx_intr_std_{seed}.csv", output["expected_errors_approx_intr_std"][1:], delimiter=',')


