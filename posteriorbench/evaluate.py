import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from posteriorbench.datasets import DATASETS, get_dataset
from posteriorbench.evaluation.metrics import (
    DEFAULT_N_PROJECTIONS,
    DEFAULT_PROJECTION_GRF_ALPHA,
    DEFAULT_PROJECTION_GRF_TAU,
    DEFAULT_PROJECTION_SEED,
    evaluate_distribution,
    normalize_weights,
)
from posteriorbench.io import write_json
from posteriorbench.training_source import open_training_field_source


METRIC_NAMES = (
    "mean_rel_l2",
    "std_rel_l2",
    "mmd",
    "swd",
    "spectral_rel_geomean",
)

_LIGHT_TRANSPORT_SIGMA_T_STATS_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def _evaluation_fields(dataset) -> tuple[str, ...]:
    if dataset.name == "light_transport":
        return ("sigma_t",)
    return tuple(dataset.target_fields)


def _average_sigma_t(fields: dict[str, np.ndarray] | h5py.File) -> np.ndarray:
    return (
        fields["sigma_t1"][:].astype(np.float32)
        + fields["sigma_t2"][:].astype(np.float32)
    ) * 0.5


def _load_prediction(path: Path, field: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if field == "sigma_t":
            missing = [
                source_field
                for source_field in ("sigma_t1", "sigma_t2")
                if source_field not in handle
            ]
            if missing:
                raise KeyError(
                    f"Fields {missing} required for 'sigma_t' not found in {path}: "
                    f"{list(handle.keys())}"
                )
            samples = _average_sigma_t(handle)
        elif field not in handle:
            raise KeyError(f"Field '{field}' not found in {path}: {list(handle.keys())}")
        else:
            samples = handle[field][:].astype(np.float32)
        if "w" in handle:
            weights = handle["w"][:].astype(np.float64)
        else:
            weights = np.ones(len(samples), dtype=np.float64)
    return samples, normalize_weights(weights, len(samples))


def _reference_samples(reference, field: str) -> np.ndarray:
    if field == "sigma_t":
        return _average_sigma_t(reference.fields)
    return reference.fields[field]


def _resolve_training_root(training_root: str | Path) -> Path:
    path = Path(training_root)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parents[1] / path


def _validate_swd_normalization_stats(
    stats: dict[str, object],
    source: str | Path,
    field: str,
) -> dict[str, object]:
    mean = float(stats["mean"])
    std = float(stats["std"])
    if not np.isfinite(mean):
        raise ValueError(f"SWD normalization mean for {field} in {source} is not finite")
    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"SWD normalization std for {field} in {source} must be positive"
        )
    return {
        "mean": mean,
        "std": std,
        "source": str(source),
        "field": field,
    }


def _metadata_swd_normalization_stats(
    training_root: str | Path,
    dataset_name: str,
    field: str,
) -> dict[str, object]:
    source = open_training_field_source(training_root, dataset_name=dataset_name)
    metadata = source.metadata
    try:
        field_stats = metadata["statistics"][field]
    except KeyError as exc:
        raise KeyError(
            f"No training statistics for {dataset_name}.{field} in {source.source}"
        ) from exc
    return _validate_swd_normalization_stats(field_stats, source.source, field)


def _light_transport_sigma_t_swd_normalization_stats(
    training_root: str | Path,
) -> dict[str, object]:
    source = open_training_field_source(training_root, dataset_name="light_transport")
    cache_key = (source.source, "sigma_t")
    cached = _LIGHT_TRANSPORT_SIGMA_T_STATS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    total = 0.0
    total_sq = 0.0
    count = 0
    batch_size = 256
    for start in range(0, source.num_samples, batch_size):
        stop = min(start + batch_size, source.num_samples)
        fields = source.read_channels(
            np.arange(start, stop, dtype=np.int64),
            ("sigma_t1", "sigma_t2"),
        ).astype(np.float64)
        values = (fields[:, 0] + fields[:, 1]) * 0.5
        total += float(np.sum(values))
        total_sq += float(np.sum(values * values))
        count += int(values.size)
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    stats = _validate_swd_normalization_stats(
        {"mean": mean, "std": float(np.sqrt(variance))},
        source.source,
        "0.5*(sigma_t1+sigma_t2)",
    )
    _LIGHT_TRANSPORT_SIGMA_T_STATS_CACHE[cache_key] = stats
    return dict(stats)


def _load_swd_normalization_stats(
    dataset_name: str,
    fields: tuple[str, ...],
    training_root: str | Path = "hf_reference_materialized/PDEFieldDataset_hf",
) -> dict[str, dict[str, object]]:
    root = _resolve_training_root(training_root)
    stats = {}
    for field in fields:
        if dataset_name == "light_transport" and field == "sigma_t":
            stats[field] = _light_transport_sigma_t_swd_normalization_stats(root)
        else:
            stats[field] = _metadata_swd_normalization_stats(root, dataset_name, field)
    return stats


def _summarize_metrics(
    per_case: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    fields = tuple(next(iter(per_case.values())))
    summary = {}
    for field in fields:
        summary[field] = {}
        for metric in METRIC_NAMES:
            values = [item[field][metric] for item in per_case.values()]
            summary[field][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated posteriors against PosteriorBench references."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--n-projections",
        type=int,
        default=DEFAULT_N_PROJECTIONS,
    )
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=DEFAULT_PROJECTION_SEED,
    )
    parser.add_argument(
        "--projection-grf-alpha",
        type=float,
        default=DEFAULT_PROJECTION_GRF_ALPHA,
    )
    parser.add_argument(
        "--projection-grf-tau",
        type=float,
        default=DEFAULT_PROJECTION_GRF_TAU,
    )
    parser.add_argument(
        "--training-source",
        default="hf_reference_materialized/PDEFieldDataset_hf",
        help="Materialized PDEFieldDataset_hf root used for SWD z-score normalization.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_started = time.perf_counter()
    wall_started = time.time()
    if args.n_projections <= 0:
        raise ValueError("--n-projections must be positive")
    if args.projection_grf_alpha <= 0:
        raise ValueError("--projection-grf-alpha must be positive")
    if args.projection_grf_tau <= 0:
        raise ValueError("--projection-grf-tau must be positive")

    dataset = get_dataset(args.dataset)
    case_paths = dataset.discover(args.cases)
    if args.case_ids:
        selected = set(args.case_ids)
        case_paths = [path for path in case_paths if path.stem in selected]
        missing = selected - {path.stem for path in case_paths}
        if missing:
            raise ValueError(f"Requested case IDs not found: {sorted(missing)}")
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise ValueError("No cases selected for evaluation")

    prediction_root = Path(args.predictions)
    evaluation_fields = _evaluation_fields(dataset)
    swd_normalization_stats = _load_swd_normalization_stats(
        args.dataset,
        evaluation_fields,
        args.training_source,
    )
    per_case: dict[str, dict[str, dict[str, float]]] = {}
    runtime_cases: dict[str, dict[str, object]] = {}
    for index, case_path in enumerate(case_paths):
        case_started = time.perf_counter()
        case = dataset.load_case(case_path)
        prediction_path = prediction_root / case.case_id / "generated.h5"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction not found: {prediction_path}")
        reference = dataset.load_reference(case_path)
        field_metrics = {}
        field_runtime = {}
        for field in evaluation_fields:
            field_started = time.perf_counter()
            predicted, pred_weights = _load_prediction(prediction_path, field)
            reference_samples = _reference_samples(reference, field)
            swd_field_stats = swd_normalization_stats[field]
            metrics = evaluate_distribution(
                predicted,
                reference_samples,
                pred_weights,
                reference.weights,
                n_projections=args.n_projections,
                projection_seed=args.projection_seed,
                projection_grf_alpha=args.projection_grf_alpha,
                projection_grf_tau=args.projection_grf_tau,
                swd_normalization_mean=swd_field_stats["mean"],
                swd_normalization_std=swd_field_stats["std"],
            )
            metrics.update(
                {
                    "n_pred": int(len(predicted)),
                    "n_reference": int(len(reference_samples)),
                }
            )
            field_metrics[field] = metrics
            field_runtime[field] = {
                "evaluation_seconds": time.perf_counter() - field_started,
                "n_pred": int(len(predicted)),
                "n_reference": int(len(reference_samples)),
            }
        per_case[case.case_id] = field_metrics
        runtime_cases[case.case_id] = {
            "total_seconds": time.perf_counter() - case_started,
            "fields": field_runtime,
        }
        print(
            f"[{index + 1}/{len(case_paths)}] "
            f"{case.case_id}: {field_metrics}"
        )

    total_seconds = time.perf_counter() - total_started
    output = {
        "dataset": args.dataset,
        "target_fields": list(evaluation_fields),
        "num_cases": len(per_case),
        "evaluation": {
            "n_projections": args.n_projections,
            "projection_seed": args.projection_seed,
            "projection_distribution": "normalized_dct_grf",
            "projection_grf_alpha": args.projection_grf_alpha,
            "projection_grf_tau": args.projection_grf_tau,
            "swd_input_normalization": {
                "type": "training_zscore",
                "scale": 1.0,
                "applies_to": ["swd"],
                "statistics": swd_normalization_stats,
            },
            "aggregate_std_ddof": 1,
            "spectral_definition": "radial_psd_per_bin_relative_error_geometric_mean",
            "spectral_rel_geomean": (
                "exp(mean(log(valid_relative_errors + 1e-10))) with "
                "valid bins requiring finite values and ref_spectrum > 1e-10"
            ),
        },
        "summary": _summarize_metrics(per_case),
        "per_case": per_case,
        "runtime": {
            "started_at_unix": wall_started,
            "total_seconds": total_seconds,
            "num_cases": len(per_case),
            "cases_per_second": (
                len(per_case) / total_seconds if total_seconds > 0 else None
            ),
            "cases": runtime_cases,
        },
    }
    output_path = Path(args.output) / "metrics.json"
    write_json(output_path, output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
