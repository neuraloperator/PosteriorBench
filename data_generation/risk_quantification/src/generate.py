"""Generate the full RiskQuantification dataset with explicit saved parameters."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path


def validate_config(c):
    from dataset_io import config_from_dict

    cfg = config_from_dict(c["solver"])
    if cfg.mode != "fixed_deadline_exit":
        raise ValueError("RiskQuantification uses fixed_deadline_exit")
    ids = c["reference_ids"]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("Invalid reference IDs")
    if len({cfg.stream, c["reference_stream"], c["training_stream"]}) != 3:
        raise ValueError("Independent streams required")
    if not 0 < c["validation_cases"] < len(ids):
        raise ValueError("Both validation and test cases required")
    for key in (
        "training_fields",
        "candidate_fields",
        "samples_per_case",
        "batch_size",
    ):
        if c[key] < 1:
            raise ValueError(f"{key} must be positive")
    if c["epsilon"] <= 0 or c["sigma"] != c["epsilon"] / 3:
        raise ValueError("Expected sigma=epsilon/3")
    if len(c["K_values"]) < 1:
        raise ValueError("K_values required")
    return cfg


def references(c, root):
    import numpy as np
    import pipeline as pp
    import risk1d as risk
    import safe_explorer_core as prior
    from dataset_io import sha256

    cfg = replace(validate_config(c), stream=c["reference_stream"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(configuration=c, numerical_source_sha256=pp.source_hash())
    marker = root / "manifest.json"
    path = root / "references.npz"
    with pp.exclusive(root / "generation.lock"):
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["contract"] != contract or saved["sha256"] != sha256(path):
                raise ValueError("Reference resume mismatch")
            return
        ids = np.asarray(c["reference_ids"], np.int64)
        p = prior.sample_parameters(ids, cfg, "moving_mix")
        u = risk.solve_parameters(p, cfg)
        risk.check_probabilities(u, cfg)
        designs = [
            prior.observation_design(
                cfg, c["K_values"][i % len(c["K_values"])], 256, c["observation_seed"]
            )
            for i in range(len(ids))
        ]
        pp.atomic_npz(
            path,
            u=u,
            masks=np.stack([d["mask"] for d in designs]),
            K=np.asarray([d["K"] for d in designs], np.int32),
            **{"parameter__" + k: v for k, v in p.items()},
        )
        pp.atomic_json(
            marker, dict(complete=True, contract=contract, sha256=sha256(path))
        )


def run(config, output, stage="all"):
    import pipeline as pp
    from pool_generation import generate
    from dataset_io import config_dict
    from export_dataset import export_dataset

    c = json.loads(Path(config).read_text())
    cfg = validate_config(c)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    contract = out / "configuration.json"
    if contract.exists() and json.loads(contract.read_text()) != c:
        raise ValueError("Configuration changed; use a new output directory")
    pp.atomic_json(contract, c)
    if stage in ("all", "references"):
        references(c, out / "references")
    if stage in ("all", "training"):
        generate(
            config_dict(replace(cfg, stream=c["training_stream"])),
            out / "training_pool",
            c["training_fields"],
            c["batch_size"],
        )
    if stage in ("all", "candidates"):
        generate(
            config_dict(cfg),
            out / "candidate_pool",
            c["candidate_fields"],
            c["batch_size"],
        )
    if stage in ("all", "export"):
        export_dataset(
            c,
            out / "training_pool",
            out / "candidate_pool",
            out / "references",
            out / "benchmark",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage",
        choices=["all", "references", "training", "candidates", "export"],
        default="all",
    )
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    run(a.config, a.output, a.stage)


if __name__ == "__main__":
    main()
