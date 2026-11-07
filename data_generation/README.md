# Data generation

## Darcy and Poisson

JAX pipeline that builds a large prior sample pool by solving the PDE, then
rejection-samples posterior ensembles for reference cases from that pool.
Run from the repository root (needs `jax`, `h5py`, `scipy`, `tqdm`; a GPU is
strongly recommended).

```text
data_generation/
  common.py                 # GRF prior, CG elliptic solver, pool + rejection loops
  darcy/   pde.py           # -div(a grad u) = 1, a in {3, 12} from a thresholded GRF
           generate_pool.py
           rejection_sampling.py
  poisson/ pde.py           # -lap(phi) = f, f ~ GRF(alpha ~ U(1.5,3), tau ~ U(2,4))
           generate_pool.py
           rejection_sampling.py
```

### 1. Prior pool

Each job writes `--files_per_job` (default 5) files of `--samples_per_file`
(default 1M) samples to `<pde>_pool_128/part_<job>_<file>.h5`. `--job_id`
selects an independent RNG stream; the original runs used job ids 0-49.

```bash
python -m data_generation.darcy.generate_pool --job_id 0
python -m data_generation.poisson.generate_pool --job_id 0
```

Datasets: Darcy `a, u`; Poisson `f, phi, alpha, tau`.

### 2. Rejection sampling

For each reference seed, a pool sample (and its 90/180/270 degree rotations)
is accepted when the RMSE between its observation and the reference
observation is at most `--threshold` (default 4e-4). Weights are
`exp(-0.5 (rmse / sigma)^2)` with `sigma = threshold / 3`, normalized to sum to 1.
A reference is kept only if `--samples_per_ref` (default 100) samples are found.

```bash
python -m data_generation.darcy.rejection_sampling --job_id 0
python -m data_generation.poisson.rejection_sampling --job_id 0
```

Observations: `--obs_mode sparse_random` (`--num_sensors` random points,
default 128) or `low_res` (`--obs_resolution` block average, default 16).
Output: `<pde>_rejection_<obs_mode>/accepted_seed_<seed>.h5` with `<field>_ref`,
the accepted `<field>` stacks, `w`, and `sensor_coords` (sparse mode).

Per-PDE defaults kept from the original runs:

| | Darcy | Poisson |
| --- | --- | --- |
| pool base seed | 888 | 777 |
| reference seeds | 1200 + 1000·job | 5800 + 1000·job |
| sensor seed | ref seed + 12345 | ref seed + 999 |
| `--refs_per_job` | 35 | 10 |
| `--chunk_size` (pool read size) | 5000 | 10000 |
| skip seed if first pool file has no match | no | yes |

## Stochastic Risk Quantification

Please see [data_generation/risk_quantification/README.md](risk_quantification/README.md) for instructions on generating the risk quantification dataset.
