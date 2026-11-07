#!/usr/bin/env python3
"""Matchability-based case discovery used by the full dataset.

Draw a reference bank from the unchanged moving prior, independently of candidates.
Fix each reference's K before matching. Retain the first ready reference IDs in each
K stratum until the requested balanced quota is filled. Never perturb reference
settings, pad ensembles, change epsilon, or select posterior samples by closeness.
The selection favors matchable observations and is reported in the manifest.
"""

from __future__ import annotations
import argparse, os, json, sys, time, hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Repeat the published matchability-based discovery stage."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int, default=8)
    cli = p.parse_args()
    c = json.loads(Path(cli.config).read_text())
    from types import SimpleNamespace

    d = c["discovery"]
    a = SimpleNamespace(
        out=cli.output,
        threads=cli.threads,
        cases=len(c["reference_ids"]),
        matches=d["matches_required"],
        reference_bank=d["reference_bank"],
        max_candidates=d["candidate_settings"],
        block_size=c["batch_size"],
        profile="fine",
        reference_stream=c["reference_stream"],
        candidate_stream=d["candidate_stream"],
    )
    if a.max_candidates % a.block_size or a.threads < 1:
        p.error("Invalid batch size or thread count")
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import numpy as np
    import pipeline as pp

    root = Path(a.out)
    spec = dict(pp.DEFAULTS)
    spec.update(
        cases=a.reference_bank,
        block_size=a.block_size,
        profile=a.profile,
        reference_stream=a.reference_stream,
        candidate_stream=a.candidate_stream,
    )
    spec.update(
        {
            k: c["solver"][k]
            for k in (
                "nx",
                "nt",
                "nz",
                "substeps",
                "domain",
                "seed",
                "max_components",
                "spatial_modes",
            )
        }
    )
    spec["observation_seed"] = c["observation_seed"]
    pp.init_run(root, spec)
    selection_path = root / "internal" / "curation.json"
    requested = dict(
        cases=a.cases,
        matches=a.matches,
        reference_bank=a.reference_bank,
        policy="First ready DISTINCT observation problems within each fixed-K quota; data-dependent, nonrepresentative.",
        quotas={
            str(k): a.cases // 7 + int(i < a.cases % 7)
            for i, k in enumerate(range(6, 13))
        },
    )
    if selection_path.exists() and json.loads(selection_path.read_text()) != requested:
        raise RuntimeError(
            "Curation target changed. Use a different run folder; old stopping rules cannot be mixed."
        )
    pp.atomic_json(selection_path, requested)
    refs = pp.load_references(root)
    # Identical masks + values define the same observation problem, even when GT differs.
    # Keep its first bank representative instead of counting duplicated queries as new cases.
    canonical = set()
    seen = set()
    for ri, (u, mask) in enumerate(zip(refs["risk"], refs["masks"])):
        key = hashlib.sha256(mask.tobytes() + u[mask].tobytes()).hexdigest()
        if key not in seen:
            canonical.add(ri)
            seen.add(key)
    pp.atomic_json(
        root / "internal" / "duplicate_observation_audit.json",
        dict(
            bank_size=len(refs["risk"]),
            distinct_observation_problems=len(canonical),
            duplicate_problem_count=len(refs["risk"]) - len(canonical),
            canonical_reference_ids=sorted(canonical),
        ),
    )

    def choose(st):
        chosen = []
        for k in range(6, 13):
            ready = [
                i
                for i, r in enumerate(st["rows"])
                if i in canonical and r["K"] == k and r["matches"] >= a.matches
            ]
            chosen += ready[: requested["quotas"][str(k)]]
        return chosen

    warm = pp.warmup(spec, refs)
    print(
        f"Curated discovery: {a.reference_bank} independent references; target {a.cases} cases x {a.matches} matches. Warmup {warm:.2f}s",
        flush=True,
    )
    ttotal = time.perf_counter()
    for start in range(0, a.max_candidates, a.block_size):
        st = pp.status(root, a.matches)
        chosen = choose(st)
        if len(chosen) >= a.cases:
            break
        stop = start + a.block_size
        path = root / "internal" / "blocks" / f"block_{start:010d}_{stop:010d}.npz"
        if path.exists():
            continue
        active = []
        for k in range(6, 13):
            ready = [
                i
                for i, r in enumerate(st["rows"])
                if i in canonical and r["K"] == k and r["ready"]
            ]
            if len(ready) < requested["quotas"][str(k)]:
                active.extend(
                    i
                    for i, r in enumerate(st["rows"])
                    if i in canonical and r["K"] == k and not r["ready"]
                )
        active = np.array(sorted(active), int)
        small = {key: refs[key][active] for key in ("risk", "K", "masks")}
        t = time.perf_counter()
        with pp.exclusive(path.with_suffix(".lock")):
            result = pp.compute_block(spec, small, start, stop)
            d = np.full((len(result["ids"]), a.reference_bank), np.inf)
            d[:, active] = result["distances"]
            result["distances"] = d
            diag = json.loads(str(result["diagnostics"]))
            diag["active_reference_indices"] = active.tolist()
            result["diagnostics"] = np.asarray(json.dumps(diag))
            pp.atomic_npz(path, **result)
        st = pp.status(root, a.matches)
        chosen = choose(st)
        print(
            f"{stop:,} settings; selected {len(chosen)}/{a.cases}; active {len(active)}; block {time.perf_counter() - t:.2f}s",
            flush=True,
        )
    st = pp.status(root, a.matches)
    chosen = choose(st)
    pp.atomic_json(
        root / "internal" / "selected_reference_ids.json",
        dict(
            chosen_reference_ids=chosen,
            complete=len(chosen) == a.cases,
            wall_search_seconds=time.perf_counter() - ttotal,
            **requested,
        ),
    )
    if len(chosen) != a.cases:
        raise RuntimeError(
            "Discovery budget exhausted without enough matchable cases; see selection file."
        )
    queues = {k: [] for k in c["K_values"]}
    for idx in chosen:
        queues[int(refs["K"][idx])].append(idx)
    ordered = [
        queues[c["K_values"][i % len(c["K_values"])]].pop(0) for i in range(a.cases)
    ]
    c["reference_ids"] = ordered
    c["discovery"]["selected_reference_ids"] = ordered
    c["discovery"]["candidate_settings"] = st["candidate_settings_processed"]
    pp.atomic_json(root / "selected_config.json", c)
    print(
        f"Saved {root / 'selected_config.json'}; use it with risk-generate.", flush=True
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed blocks preserved.", file=sys.stderr)
        sys.exit(130)
