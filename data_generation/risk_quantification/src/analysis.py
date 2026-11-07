"""Posterior summaries, sample overlap, and prediction evaluation."""

from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from reference_metrics import (
    ensemble,
    moments,
    compare,
    grf_directions,
    METRICS,
)


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def cases(root):
    from datasets import load_from_disk

    ds = load_from_disk(
        str(Path(root) / "PosteriorDataset_hf/RiskQuantification")
    ).with_format("numpy", dtype=None)
    for split in ("validation", "test"):
        if split in ds:
            for row in ds[split]:
                yield split, row


def summary(root, out):
    import hashlib

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sets = []
    rows = []
    for split, row in cases(root):
        fields, w = ensemble(row["posterior_samples_u"], row["posterior_weights"])
        mean, std = moments(fields, w)
        cid = str(row["metadata"]["case_id"])
        np.savez_compressed(out / f"{cid}_moments.npz", mean=mean, std=std)
        # Exact float32 field hashes also detect distinct IDs with identical values.
        hashes = {
            hashlib.sha256(
                np.ascontiguousarray(f, dtype=np.float32).tobytes()
            ).hexdigest()
            for f in fields
        }
        sets.append(hashes)
        rows.append(
            dict(
                case_id=cid,
                split=split,
                samples=len(fields),
                unique_samples=len(hashes),
                ess=float(1 / (w @ w)),
            )
        )
    matrix = np.array([[len(a & b) for b in sets] for a in sets])
    upper = matrix[np.triu_indices(len(sets), 1)]
    reuse = Counter(x for s in sets for x in s)
    report = dict(
        cases=rows,
        cross_case_overlap_matrix=matrix.tolist(),
        pairwise_histogram={str(k): int(v) for k, v in sorted(Counter(upper).items())},
        unique_fields=len(reuse),
        unique_field_multiplicity={
            str(k): int(v) for k, v in sorted(Counter(reuse.values()).items())
        },
        maximum_pairwise_overlap=int(upper.max()) if len(upper) else 0,
    )
    write_json(out / "summary.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    display = matrix.astype(float)
    np.fill_diagonal(display, np.nan)
    im = axes[0].imshow(display, cmap="Blues")
    fig.colorbar(im, ax=axes[0], label="Shared samples")
    axes[0].set(title="Cross-case overlap", xlabel="Case index", ylabel="Case index")
    if len(upper):
        axes[1].hist(upper, bins=np.arange(upper.max() + 2) - 0.5)
    axes[1].set(
        title="Pairwise overlap distribution",
        xlabel="Shared samples",
        ylabel="Case pairs",
    )
    fig.savefig(out / "overlap.png", dpi=150)
    plt.close(fig)
    return report


def evaluate(root, predictions, out, split="test"):
    """Evaluate saved predictions against the dataset's reference ensembles."""
    meta = json.loads(
        (Path(root) / "PDEFieldDataset_hf/RiskQuantification/metadata.json").read_text()
    )
    norm = meta["statistics"]["u"]
    normalization = dict(mean=norm["mean"], std=norm["std"])
    primary = {str(r["metadata"]["case_id"]): r for s, r in cases(root) if s == split}
    rows = []
    directions = None
    for cid, r in primary.items():
        if directions is None:
            directions = grf_directions(r["u_ref"].shape)
        with np.load(Path(predictions) / f"{cid}.npz", allow_pickle=False) as z:
            u = z["u"]
            w = z["weights"] if "weights" in z.files else None
        a = compare(
            u,
            r["posterior_samples_u"],
            directions,
            normalization,
            wa=w,
            wb=r["posterior_weights"],
        )
        rows.append(dict(case_id=cid, reference1=a))
    if not rows:
        raise ValueError("No cases for requested split")
    averages = {
        "reference1": {
            metric: float(np.mean([r["reference1"][metric] for r in rows]))
            for metric in METRICS
        }
    }
    report = dict(
        split=split,
        cases=rows,
        average_scores=averages,
        normalization=normalization,
        convention="Weighted paper-based diagnostics; not the official PosteriorBench CLI",
    )
    write_json(out, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dataset", required=True)
    s.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", required=True)
    e.add_argument("--predictions", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    a = p.parse_args()
    if a.command == "summary":
        summary(a.dataset, a.output)
    else:
        evaluate(a.dataset, a.predictions, a.output, split=a.split)


if __name__ == "__main__":
    main()
