"""Plot saved physical fields or reference posterior inference; no RNG or solver."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from dataset_io import load_sample, read_observations
from reference_metrics import ensemble, moments
from risk1d import evaluate_settings


def plot_sample(root, output, **selection):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, params, cfg, label = load_sample(root, **selection)
    tau = np.linspace(0, 1, cfg.nt)
    x = np.linspace(*cfg.domain, cfg.nx)
    # Align chronological geometry/dynamics to the risk field's remaining-time axis.
    values = evaluate_settings(params, cfg, times=1 - tau, x=x)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
    bound = max(float(np.nanmax(np.abs(values["drift"]))), 1e-12)
    panels = [
        (u, "Risk probability u", "viridis", 0, 1),
        (values["safe_mask"], "Safe set (open intervals)", "Greys", 0, 1),
        (values["drift"], "Physical drift b(x,t)", "RdBu_r", -bound, bound),
        (values["diffusivity"], "Physical diffusivity D(x,t)", "magma", 0, None),
    ]
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(*cfg.domain, 0, 1),
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set(title=title, xlabel="Position x", ylabel="Remaining time τ = 1 − t")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for c in range(int(params["ncomp"][0])):
            for side in (0, 1):
                ax.plot(
                    values["safe_intervals"][:, c, side], tau, color="#e95420", lw=0.8
                )
    fig.suptitle(
        label
        + "\nSaved parameters; dynamics outside the safe set are undefined (blank).",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--view", choices=["physical", "inference"], default="physical")
    p.add_argument(
        "--collection",
        choices=["reference", "posterior", "training", "candidate"],
        default="reference",
    )
    p.add_argument("--case", default="case_0000")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.view == "inference":
        if a.collection != "reference":
            p.error("--view inference summarizes a case; omit --collection")
        plot_inference(a.dataset, a.output, a.case)
    else:
        plot_sample(
            a.dataset,
            a.output,
            collection=a.collection,
            case_id=a.case,
            sample_index=a.sample_index,
            sample_id=a.sample_id,
        )
    print(a.output)


def plot_inference(root, output, case_id="case_0000"):
    """Display observations and population moments using saved posterior weights."""
    from datasets import load_from_disk
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    obs = read_observations(root, case_id)
    manifest = json.loads((root / "manifest.json").read_text())
    case = next(c for c in manifest["case_index"] if c["case_id"] == case_id)
    dataset = load_from_disk(str(root / "PosteriorDataset_hf/RiskQuantification"))
    row = dataset[case["split"]].with_format("numpy", dtype=None)[case["row"]]
    if row["metadata"]["case_id"] != case_id:
        raise ValueError("Case mapping mismatch")
    fields, weights = ensemble(row["posterior_samples_u"], row["posterior_weights"])
    mean, std = moments(fields, weights)
    observed = np.full(mean.shape, np.nan)
    coords = obs["sensor_coords"]
    observed[coords[:, 0], coords[:, 1]] = obs["observation_values"]
    panels = [
        (row["u_ref"], "Reference risk field", "viridis", 1.0),
        (observed, f"Observed risk ({len(coords)} pixels)", "viridis", 1.0),
        (mean, "Reference posterior mean", "viridis", 1.0),
        (std, "Reference posterior std.", "magma", max(float(std.max()), 1e-12)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
    for ax, (values, title, palette, maximum) in zip(axes, panels):
        cmap = plt.get_cmap(palette).copy()
        cmap.set_bad("white")
        im = ax.imshow(
            np.ma.masked_invalid(values),
            origin="lower",
            aspect="auto",
            extent=(obs["x"][0], obs["x"][-1], obs["tau"][0], obs["tau"][-1]),
            cmap=cmap,
            vmin=0.0,
            vmax=maximum,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="Position x", ylabel="Remaining time τ = 1 − t")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{case_id} · {len(fields)} weighted reference posterior samples\n"
        "Stored observations and reference ensemble; no trained-model predictions.",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return dict(mean=mean, std=std, observations=observed, weights=weights)


if __name__ == "__main__":
    main()
