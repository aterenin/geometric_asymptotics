#!/usr/bin/env python

from setuptools import find_namespace_packages, setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "jax>=0.4.14",
    "jaxlib>=0.4.14",
    "gmshparser>=0.2.0",
    "matplotlib>=3.7.2",
    "plotly>=5.16.1",
    "ipywidgets>=8.1.1",
    "optax==0.1.7",
    "jaxkern==0.0.5",
    "gpjax==0.5.9",
    "GeometricKernels @ git+https://github.com/GPflow/GeometricKernels@d5f39b7882a859b8bf7d51fe259baea264b5e5dd",
]

setup(name='geometric_asymptotics',
      version='1.0',
      description='Experiments for geometric asymptotics',
      author='Alexander Terenin and collaborators',
      install_requires=requirements,
      packages=find_namespace_packages(include=["geometric_asymptotics*"]),
     )