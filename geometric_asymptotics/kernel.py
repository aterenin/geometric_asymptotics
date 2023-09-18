import jax.numpy as jnp
from jaxkern.base import AbstractKernel
from jaxkern.computations import AbstractKernelComputation
from geometric_kernels.frontends.jax.gpjax import _GeometricComputation
from typing import Dict, Optional, List
from jax.random import KeyArray
from jaxtyping import Float, Array
from jaxutils.config import get_global_config
get_global_config().transformations["nu"] = "identity_transform"

class ScaleKernel(AbstractKernel):
    def __init__(
        self,
        kernel: AbstractKernel,
        compute_engine: AbstractKernelComputation = _GeometricComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Scale Kernel",
    ) -> None:
        self.kernel = kernel
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return params['variance'] * self.kernel(params, x, y)

    def init_params(self, key: KeyArray) -> Dict:
        params = self.kernel.init_params(key)
        assert not 'variance' in params
        params['variance'] = jnp.array([1.0])
        return params