"""Shared pieces of the Darcy/Poisson data generation: GRF prior, elliptic solver,
pool generation and rejection sampling."""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import glob
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import idctn
from jax.scipy.sparse.linalg import cg
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

S = 128


@partial(jax.jit, static_argnums=(2,))
def grf(alpha, tau, s, key):
    """Zero-mean Gaussian random field on an s x s grid (spectral DCT construction)."""
    xi = jax.random.normal(key, (s, s))
    k1, k2 = jnp.meshgrid(jnp.arange(s), jnp.arange(s))
    coef = (tau ** (alpha - 1)) * (jnp.pi**2 * (k1**2 + k2**2) + tau**2) ** (-alpha / 2)
    L = (s * coef * xi).at[0, 0].set(0.0)
    return idctn(L, type=2, norm="ortho")


def _interp_matrix(K, to_node):
    """Cubic-spline interpolation matrix between cell-center and node grids."""
    coords_center = np.linspace(1 / (2 * K), (2 * K - 1) / (2 * K), K)
    coords_node = np.linspace(0, 1, K)
    old, new = (coords_center, coords_node) if to_node else (coords_node, coords_center)
    I = np.eye(K)
    M = np.zeros((K, K))
    for i in range(K):
        z = np.tile(I[:, i : i + 1], (1, 2))
        sp = RectBivariateSpline(old, np.array([0, 1]), z, kx=3, ky=1)
        M[:, i] = sp(new, np.array([0.5])).flatten()
    return jnp.array(M)


M_TO_NODE = _interp_matrix(S, to_node=True)
M_TO_NODE_T = M_TO_NODE.T
M_TO_CENTER = _interp_matrix(S, to_node=False)
M_TO_CENTER_T = M_TO_CENTER.T


def to_node(x):
    return M_TO_NODE @ x @ M_TO_NODE_T


def solve_elliptic(coef_v, F_raw):
    """Solve -div(coef grad p) = F with zero Dirichlet BC by CG.

    coef_v is given on nodes; F_raw and the returned p live on cell centers.
    """
    n = S - 2
    dx_inv_sq = (S - 1) ** 2
    f_vec = to_node(F_raw)[1:-1, 1:-1].flatten(order="F")

    cs = (coef_v[:-2, 1:-1] + coef_v[1:-1, 1:-1]) / 2.0
    cn = (coef_v[2:, 1:-1] + coef_v[1:-1, 1:-1]) / 2.0
    cw = (coef_v[1:-1, :-2] + coef_v[1:-1, 1:-1]) / 2.0
    ce = (coef_v[1:-1, 2:] + coef_v[1:-1, 1:-1]) / 2.0
    diag = (cw + ce + cs + cn) * dx_inv_sq

    def operator(p_flat):
        p = p_flat.reshape((n, n), order="F")
        p_s = jnp.vstack([jnp.zeros((1, n)), p[:-1, :]])
        p_n = jnp.vstack([p[1:, :], jnp.zeros((1, n))])
        p_w = jnp.hstack([jnp.zeros((n, 1)), p[:, :-1]])
        p_e = jnp.hstack([p[:, 1:], jnp.zeros((n, 1))])
        val = diag * p - (cs * p_s + cn * p_n + cw * p_w + ce * p_e) * dx_inv_sq
        return val.flatten(order="F")

    p_vec, _ = cg(operator, f_vec, tol=1e-14, maxiter=10000)
    P_node = jnp.pad(p_vec.reshape((n, n), order="F"), ((1, 1), (1, 1)), mode="constant")
    return M_TO_CENTER @ P_node @ M_TO_CENTER_T


def generate_pool(sample, pde, base_seed):
    """CLI entry: write `files_per_job` HDF5 files of prior samples drawn by `sample(key) -> dict`."""
    parser = argparse.ArgumentParser(description=f"Generate the {pde} prior sample pool.")
    parser.add_argument("--job_id", type=int, default=0, help="Selects the RNG stream and output file names.")
    parser.add_argument("--out_dir", default=f"{pde}_pool_{S}")
    parser.add_argument("--files_per_job", type=int, default=5)
    parser.add_argument("--samples_per_file", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    print("JAX devices:", jax.devices())
    os.makedirs(args.out_dir, exist_ok=True)
    B = args.batch_size
    n_batches = len(range(0, args.samples_per_file, B))
    sample_batch = jax.jit(jax.vmap(sample))
    key = jax.random.fold_in(jax.random.PRNGKey(base_seed), args.job_id)

    for file_idx in range(args.files_per_job):
        tag = f"part_{args.job_id:03d}_{file_idx:02d}"
        path = os.path.join(args.out_dir, f"{tag}.h5")
        with h5py.File(path, "w") as h5:
            dsets = None
            for i in tqdm(range(n_batches), desc=f"Job {args.job_id} [{tag}]"):
                key, *subkeys = jax.random.split(key, B + 1)
                batch = sample_batch(jnp.array(subkeys))
                if dsets is None:
                    dsets = {
                        name: h5.create_dataset(
                            name, shape=(n_batches * B, *v.shape[1:]), dtype="float32",
                            **(dict(chunks=(B, S, S), compression="gzip", compression_opts=4) if v.ndim == 3 else {}),
                        )
                        for name, v in batch.items()
                    }
                for name, v in batch.items():
                    dsets[name][i * B : (i + 1) * B] = np.asarray(v, dtype=np.float32)
        print(f"Finished {path}")


def sensor_indices(num_sensors, seed):
    idx = np.random.default_rng(seed).choice(S * S, size=num_sensors, replace=False)
    return idx // S, idx % S


def low_res(field, res):
    """Block-average the last two axes to res x res, then upsample back by repetition."""
    if res == S:
        return field
    f = S // res
    coarse = field.reshape(*field.shape[:-2], res, f, res, f).mean(axis=(-3, -1))
    return np.repeat(np.repeat(coarse, f, axis=-1), f, axis=-2)


def obs_rmse(obs, ref_obs):
    return np.sqrt(np.mean((obs - ref_obs) ** 2, axis=tuple(range(1, obs.ndim))))


def collect_posterior(ref, observe, observed, pool_files, n_target, threshold, sigma, chunk_size, early_exit, pbar):
    """Scan the pool (with the 4 rotations of every chunk) for samples whose observation is
    within `threshold` RMSE of the reference. Returns (fields, weights) or None if fewer than
    `n_target` are found."""
    ref_obs = observe(ref[observed])
    accepted = {name: [] for name in ref}
    errors = []
    n = 0
    for file_idx, path in enumerate(pool_files):
        with h5py.File(path, "r") as h5:
            num = h5[observed].shape[0]
            for start in range(0, num, chunk_size):
                chunk = {name: h5[name][start : min(start + chunk_size, num)] for name in ref}
                for k in range(4):
                    rot = {name: np.rot90(v, k=k, axes=(1, 2)) if v.ndim == 3 else v for name, v in chunk.items()}
                    err = obs_rmse(observe(rot[observed]), ref_obs)
                    idx = np.where(err <= threshold)[0][: n_target - n]
                    for name in ref:
                        accepted[name].append(rot[name][idx])
                    errors.append(err[idx])
                    n += len(idx)
                    pbar.n = n
                    pbar.refresh()
                    if n >= n_target:
                        break
                if n >= n_target:
                    break
        if n >= n_target or (early_exit and file_idx == 0 and n == 0):
            break
    if n < n_target:
        return None
    fields = {name: np.concatenate(v).astype("float32") for name, v in accepted.items()}
    weights = np.exp(-0.5 * (np.concatenate(errors) / sigma) ** 2).astype("float32")
    return fields, weights


def rejection_sampling(sample, observed, pde, *, ref_seed_start, sensor_seed_offset, refs_per_job, chunk_size, early_exit):
    """CLI entry: draw reference cases from `sample` and collect pool samples consistent with
    their observation of field `observed`."""
    parser = argparse.ArgumentParser(description=f"Rejection-sample {pde} posteriors from the prior pool.")
    parser.add_argument("--job_id", type=int, default=0, help="Selects the reference seed range.")
    parser.add_argument("--pool_dir", default=f"{pde}_pool_{S}")
    parser.add_argument("--out_dir", default=None, help=f"Default: {pde}_rejection_<obs_mode>")
    parser.add_argument("--obs_mode", choices=["sparse_random", "low_res"], default="sparse_random")
    parser.add_argument("--num_sensors", type=int, default=128, help="sparse_random only")
    parser.add_argument("--obs_resolution", type=int, default=16, help="low_res only")
    parser.add_argument("--threshold", type=float, default=4e-4, help="Observation RMSE threshold; sigma = threshold / 3.")
    parser.add_argument("--samples_per_ref", type=int, default=100)
    parser.add_argument("--refs_per_job", type=int, default=refs_per_job)
    parser.add_argument("--ref_seed_start", type=int, default=ref_seed_start)
    parser.add_argument("--chunk_size", type=int, default=chunk_size)
    args = parser.parse_args()

    out_dir = args.out_dir or f"{pde}_rejection_{args.obs_mode}"
    os.makedirs(out_dir, exist_ok=True)
    pool_files = sorted(glob.glob(os.path.join(args.pool_dir, "*.h5")))
    if not pool_files:
        raise FileNotFoundError(f"No pool files in {args.pool_dir}")
    sigma = args.threshold / 3

    seed = args.ref_seed_start + args.job_id * 1000
    done = 0
    pbar = tqdm(total=args.samples_per_ref, unit="sample", dynamic_ncols=True)
    while done < args.refs_per_job:
        pbar.reset(total=args.samples_per_ref)
        pbar.set_description(f"Job {args.job_id} | Seed {seed}")
        pbar.set_postfix_str(f"Overall: {done}/{args.refs_per_job}")

        ref = {name: np.asarray(v) for name, v in sample(jax.random.PRNGKey(seed)).items()}
        if args.obs_mode == "sparse_random":
            ox, oy = sensor_indices(args.num_sensors, seed + sensor_seed_offset)
            observe = lambda u: u[..., ox, oy]
        else:
            observe = lambda u: low_res(u, args.obs_resolution)

        result = collect_posterior(ref, observe, observed, pool_files, args.samples_per_ref,
                                   args.threshold, sigma, args.chunk_size, early_exit, pbar)
        if result is not None:
            fields, w = result
            w_sum = np.sum(w)
            with h5py.File(os.path.join(out_dir, f"accepted_seed_{seed}.h5"), "w") as h5:
                for name, v in ref.items():
                    h5.create_dataset(f"{name}_ref", data=v.astype("float32"))
                if args.obs_mode == "sparse_random":
                    h5.create_dataset("sensor_coords", data=np.stack([ox, oy], axis=1).astype("int32"))
                for name, v in fields.items():
                    h5.create_dataset(name, data=v)
                dset_w = h5.create_dataset("w", data=w / w_sum if w_sum > 0 else w)
                if w_sum > 0:
                    dset_w.attrs["original_sum"] = w_sum
            done += 1
            tqdm.write(f"Job {args.job_id}: finished seed {seed} ({done}/{args.refs_per_job})")
        seed += 1
    pbar.close()
    print(f"Job {args.job_id}: all {args.refs_per_job} references collected.")
