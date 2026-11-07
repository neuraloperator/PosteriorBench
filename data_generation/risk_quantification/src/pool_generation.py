"""Generate a new u-only pool, saving the exact parameters supplied to the solver.

Use the same command for candidates or independent training fields, with separate
configs/streams. Completed batches can be resumed without generating them again.
"""

import json
from pathlib import Path
import numpy as np
import pipeline as pp
from dataset_io import (
    config_from_dict,
    config_dict,
    sha256,
    solve_saved_batch,
    environment_info,
)


def generate(config, out, count, batch_size=256):
    cfg = config_from_dict(config)
    out = Path(out)
    if count < 1 or batch_size < 1:
        raise ValueError("Counts must be positive")
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(
        format_version=2,
        count=count,
        batch_size=batch_size,
        solver_config=config_dict(cfg),
        numerical_source_sha256=pp.source_hash(),
        prior="moving_mix",
    )
    path = out / "generation_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Generation contract changed")
    if (out / "conversion_contract.json").exists():
        raise ValueError("Cannot generate into a migrated pool")
    pp.atomic_json(path, contract)
    records = []
    environment = environment_info()
    with pp.exclusive(out / "generation.lock"):
        pp.atomic_json(out / "manifest.json", dict(format_version=2, complete=False))
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            ids = np.arange(start, stop, dtype=np.int64)
            path = out / "blocks" / f"fields_{start:010d}_{stop:010d}.npz"
            audit = path.with_suffix(".json")
            if path.exists() and audit.exists():
                record = json.loads(audit.read_text())
                if sha256(path) != record["sha256"]:
                    raise ValueError("Corrupt generated block")
            else:
                solve_saved_batch(path, ids, cfg)
                record = dict(
                    file=str(path.relative_to(out)),
                    start=start,
                    stop=stop,
                    count=len(ids),
                    sha256=sha256(path),
                    origin="exact_parameters_passed_to_solver",
                    environment=environment,
                )
                pp.atomic_json(audit, record)
            records.append(record)
            pp.atomic_json(
                out / "progress.json",
                dict(completed=stop, target=count, complete=stop == count),
            )
            print(f"{stop}/{count}", flush=True)
        pp.atomic_json(
            out / "manifest.json",
            dict(
                format_version=2,
                complete=True,
                kind="candidate_pool",
                field_name="u",
                field_count=count,
                solver_config=config_dict(cfg),
                blocks=records,
                prior="moving_mix",
                parameter_origin="exact_parameters_passed_to_solver",
                numerical_source_sha256=pp.source_hash(),
                environment=environment,
            ),
        )
