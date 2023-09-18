# Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds

<table>
<tr>
<td>
<img alt="Dumbbell kernel" src="https://raw.githubusercontent.com/aterenin/geometric_asymptotics/main/plot/db_kernel_intr.png">
</td>
<td>
<img alt="Sphere sample" src="https://raw.githubusercontent.com/aterenin/geometric_asymptotics/main/plot/s2_sample_intr.png">
</td>
<td>
<img alt="Dragon manifold kernel" src="https://raw.githubusercontent.com/aterenin/geometric_asymptotics/main/plot/dr_kernel_intr.png">
</td>
</tr>
</table>

## Experiments

This repository contains the experiment code for *Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds* by Paul Rosa, Viacheslav Borovitskiy, Alexander Terenin, and Judith Rousseau. The code was written by Alexander Terenin.

Usage:
```
python scripts/asymptotics.py experiment.toml
```

There are three experiment configurations available: `dumbbell.toml`, `sphere.toml`, and `dragon.toml`. 
There are also three notebooks, one for each space. 
These provide interactive visualizations of the computations performed as part of the experiment.
