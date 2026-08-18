import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from posteriorbench.datasets.base import BenchmarkCase


SCHEMA_VERSION = "posteriorbench.generated.v1"


def write_generated_posterior(
    path: str | Path,
    samples: dict[str, np.ndarray],
    case: BenchmarkCase,
    method: str,
    checkpoint: str | Path,
    profile: str | Path,
    seed: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_counts = {len(values) for values in samples.values()}
    if len(sample_counts) != 1:
        raise ValueError("All generated fields must contain the same number of samples")
    num_samples = sample_counts.pop()

    with h5py.File(output_path, "w") as handle:
        for field, values in samples.items():
            handle.create_dataset(
                field,
                data=np.asarray(values, dtype=np.float32),
                compression="gzip",
                chunks=True,
            )
        handle.create_dataset(
            "w",
            data=np.full(num_samples, 1.0 / num_samples, dtype=np.float32),
        )
        handle.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case.case_id,
                "dataset": case.dataset,
                "method": method,
                "target_fields": json.dumps(case.target_fields),
                "observed_field": case.observed_field,
                "checkpoint": str(Path(checkpoint)),
                "profile": str(Path(profile)),
                "seed": int(seed),
            }
        )
        if len(case.target_fields) == 1:
            handle.attrs["target_field"] = case.target_field


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
