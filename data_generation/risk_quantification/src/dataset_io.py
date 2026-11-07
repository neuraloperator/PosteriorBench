"""Portable single-field RiskQuantification format and explicit parameter I/O.

Loading/visualizing this format never calls the random parameter generator.
Parameter coefficients are authoritative; seeds are provenance only.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

import pipeline as pp
import risk1d as risk

FORMAT_VERSION = 2


def environment_info():
    """Record the execution environment with generated data, not as a public report."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "llvmlite", "datasets", "pyarrow")
        },
    }


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def config_dict(cfg):
    return json.loads(json.dumps(asdict(cfg)))


def config_from_dict(value):
    value = dict(value)
    value["domain"] = tuple(value["domain"])
    return risk.Config(**value).validate()


def parameter_arrays(z, prefix="parameter__"):
    return {k[len(prefix) :]: np.asarray(z[k]) for k in z.files if k.startswith(prefix)}


def save_parameters(path, params, cfg, origin):
    """Persist parameters without generating them; used by future solver runs too."""
    ids = np.asarray(params["sample_id"])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate parameter sample IDs")
    if any(len(v) != len(ids) for v in params.values()):
        raise ValueError("Parameter row counts differ")
    pp.atomic_npz(path, **params)
    with np.load(path, allow_pickle=False) as z:
        for k, v in params.items():
            np.testing.assert_array_equal(z[k], v)
            assert z[k].dtype == v.dtype
    record = dict(
        format_version=FORMAT_VERSION,
        count=len(ids),
        stream=cfg.stream,
        config=config_dict(cfg),
        origin=origin,
        sha256=sha256(path),
    )
    pp.atomic_json(Path(path).with_suffix(".json"), record)
    return record


def solve_saved_batch(path, ids, cfg):
    """Future generation: save the exact parameters used for the solved fields."""
    import safe_explorer_core as prior

    params = prior.sample_parameters(ids, cfg, "moving_mix")
    u = risk.solve_parameters(params, cfg)
    risk.check_probabilities(u, cfg)
    pp.atomic_npz(
        path,
        u=u,
        sample_ids=params["sample_id"],
        **{"parameter__" + k: v for k, v in params.items()},
    )
    return u, params


def _one(params, index):
    return {k: v[index : index + 1] for k, v in params.items()}


def _lookup(records, sample_id):
    for r in records:
        if r["start"] <= sample_id < r["stop"]:
            return r
    raise KeyError(f"Sample ID not stored: {sample_id}")


def read_observations(root, case_id):
    """Inference-only result, selecting metadata without loading full field columns."""
    from datasets import load_from_disk

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"] or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Expected a complete dataset")
    case = next(c for c in manifest["case_index"] if c["case_id"] == case_id)
    ds = load_from_disk(str(root / "PosteriorDataset_hf/RiskQuantification"))
    meta = ds[case["split"]].select_columns(["metadata"])[case["row"]]["metadata"]
    coords = np.asarray(meta["sensor_coords"], np.int32)
    values = np.asarray(meta["observation_values"], np.float32)
    x = np.asarray(meta["x_grid"], float)
    tau = np.asarray(meta["tau_grid"], float)
    if coords.shape != (256, 2) or values.shape != (256,):
        raise ValueError("Expected 256 observations")
    if (
        (coords < 0).any()
        or (coords[:, 0] >= len(tau)).any()
        or (coords[:, 1] >= len(x)).any()
    ):
        raise ValueError("Invalid observation coordinates")
    mask = np.zeros((len(tau), len(x)), bool)
    mask[coords[:, 0], coords[:, 1]] = True
    if mask.sum() != 256 or mask[0].any():
        raise ValueError("Expected unique observations excluding tau=0")
    return dict(
        case_id=case_id,
        observation_values=values,
        sensor_coords=coords,
        observation_mask=mask,
        x=x,
        tau=tau,
    )


def load_sample(
    root, *, collection="reference", case_id="case_0000", sample_index=0, sample_id=0
):
    """Return (u, parameters, Config, label), reading only explicit saved arrays.

    collection: reference/posterior/training for a release, candidate for a pool.
    posterior sample_index follows the stored weight order, not candidate ID order.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format_version"] != FORMAT_VERSION or not manifest["complete"]:
        raise ValueError("Expected a completed dataset")
    if collection == "candidate":
        record = _lookup(manifest["blocks"], sample_id)
        with np.load(root / record["file"], allow_pickle=False) as z:
            pos = np.flatnonzero(z["sample_ids"] == sample_id)
            if len(pos) != 1:
                raise ValueError("Candidate ID mapping mismatch")
            i = int(pos[0])
            u = z["u"][i]
            p = _one(parameter_arrays(z), i)
        cfg = config_from_dict(manifest["solver_config"])
        return u, p, cfg, f"Candidate {sample_id} (stream {cfg.stream})"
    from datasets import load_from_disk

    if collection == "training":
        record = _lookup(manifest["training_parameters"], sample_id)
        with np.load(root / record["file"], allow_pickle=False) as z:
            pos = np.flatnonzero(z["sample_id"] == sample_id)
            if len(pos) != 1:
                raise ValueError("Training ID mapping mismatch")
            p = _one({k: z[k] for k in z.files}, int(pos[0]))
        ds = load_from_disk(
            str(root / "PDEFieldDataset_hf/RiskQuantification")
        ).with_format("numpy", dtype=None)
        row = ds[sample_id]
        if row["metadata"]["field_id"] != f"field_{sample_id:06d}":
            raise ValueError("Training row mapping mismatch")
        cfg = config_from_dict(manifest["training_config"])
        return row["u"], p, cfg, f"Training {sample_id} (stream {cfg.stream})"
    if collection not in ("reference", "posterior"):
        raise ValueError(f"Unknown collection: {collection}")
    case = next(c for c in manifest["case_index"] if c["case_id"] == case_id)
    ds = load_from_disk(
        str(root / "PosteriorDataset_hf/RiskQuantification")
    ).with_format("numpy", dtype=None)
    row = ds[case["split"]][case["row"]]
    if row["metadata"]["case_id"] != case_id:
        raise ValueError("Case mapping mismatch")
    with np.load(root / case["parameters"], allow_pickle=False) as z:
        prefix = "reference__" if collection == "reference" else "posterior__"
        p = parameter_arrays(z, prefix)
    cfg = config_from_dict(
        manifest["reference_config"]
        if collection == "reference"
        else manifest["candidate_config"]
    )
    if collection == "reference":
        u = row["u_ref"]
        if int(p["sample_id"][0]) != case["reference_id"]:
            raise ValueError("Reference mapping mismatch")
        label = f"{case_id}: reference (bank ID {case['reference_id']})"
    else:
        if not 0 <= sample_index < len(row["posterior_weights"]):
            raise IndexError("Posterior sample index out of bounds")
        p = _one(p, sample_index)
        u = row["posterior_samples_u"][sample_index]
        if int(p["sample_id"][0]) != case["posterior_sample_ids"][sample_index]:
            raise ValueError("Posterior mapping mismatch")
        label = f"{case_id}: posterior {sample_index} (ID {int(p['sample_id'][0])})"
    return u, p, cfg, label
