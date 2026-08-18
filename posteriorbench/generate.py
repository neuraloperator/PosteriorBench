import argparse
import json
import time

import numpy as np
from pathlib import Path

from posteriorbench.artifacts import load_checkpoint_profile
from posteriorbench.datasets import DATASETS, get_dataset
from posteriorbench.io import write_generated_posterior, write_json
from posteriorbench.methods import METHODS, create_method
from posteriorbench.visualization import (
    DEFAULT_VISUALIZATION_SEED,
    write_posterior_samples_figure,
)


def validate_profile_channels(method: str, dataset, channels: tuple[str, ...]) -> None:
    if method == "funddps":
        from posteriorbench.methods.funddps import validate_funddps_channels

        validate_funddps_channels(
            dataset.name,
            dataset.target_fields,
            channels,
        )
        return
    if method == "ddis":
        from posteriorbench.methods.ddis import validate_ddis_channels

        validate_ddis_channels(
            dataset.name,
            dataset.target_fields,
            channels,
        )
        return
    if method == "fundiff":
        from posteriorbench.methods.fundiff import validate_fundiff_channels

        validate_fundiff_channels(
            dataset.name,
            dataset.target_fields,
            channels,
        )
        return
    if method == "mcdropout":
        from posteriorbench.methods.mcdropout import validate_mcdropout_channels

        validate_mcdropout_channels(
            dataset.name,
            dataset.target_fields,
            channels,
        )
        return
    dataset.validate_model_channels(channels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate posterior samples for PosteriorBench cases."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument(
        "--cases",
        required=True,
        help="PosteriorDataset_hf root or dataset-specific materialized directory",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True, help="Checkpoint profile YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=DEFAULT_VISUALIZATION_SEED,
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cases/profile and print the worklist without loading a checkpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_started = time.perf_counter()
    wall_started = time.time()
    dataset = get_dataset(args.dataset)
    profile = load_checkpoint_profile(args.profile)
    if profile.method != args.method:
        raise ValueError(
            f"Profile method '{profile.method}' does not match --method '{args.method}'"
        )
    if profile.dataset != args.dataset:
        raise ValueError(
            f"Profile dataset '{profile.dataset}' does not match --dataset '{args.dataset}'"
        )
    validate_profile_channels(args.method, dataset, profile.channels)

    case_paths = dataset.discover(args.cases)
    if args.case_ids:
        selected = set(args.case_ids)
        case_paths = [path for path in case_paths if path.stem in selected]
        missing = selected - {path.stem for path in case_paths}
        if missing:
            raise ValueError(f"Requested case IDs not found: {sorted(missing)}")
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]

    output_root = Path(args.output)
    run_config = {
        "dataset": args.dataset,
        "method": args.method,
        "cases": str(args.cases),
        "checkpoint": str(args.checkpoint),
        "profile": str(Path(args.profile)),
        "output": str(output_root),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "visualization_seed": args.visualization_seed,
        "case_ids": [path.stem for path in case_paths],
    }
    print(json.dumps(run_config, indent=2))
    if args.dry_run:
        for path in case_paths:
            dataset.load_case(path)
        print(f"Dry run validated {len(case_paths)} case(s).")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", run_config)
    method = create_method(
        args.method,
        checkpoint=args.checkpoint,
        profile=profile,
        dataset=dataset,
        device=args.device,
    )
    load_started = time.perf_counter()
    method.load()
    load_seconds = time.perf_counter() - load_started

    runtime: dict[str, object] = {
        "started_at_unix": wall_started,
        "load_seconds": load_seconds,
        "num_samples_requested_per_case": args.num_samples,
        "batch_size": args.batch_size,
    }
    runtime_cases: dict[str, dict[str, object]] = {}
    generated_cases = 0
    skipped_cases = 0
    total_generation_seconds = 0.0

    for index, case_path in enumerate(case_paths):
        case = dataset.load_case(case_path)
        output_path = output_root / case.case_id / "generated.h5"
        if output_path.exists() and not args.overwrite:
            print(f"[{index + 1}/{len(case_paths)}] Skipping {case.case_id}")
            skipped_cases += 1
            runtime_cases[case.case_id] = {"skipped": True}
            continue
        print(f"[{index + 1}/{len(case_paths)}] Generating {case.case_id}")
        case_seed = args.seed + index
        case_started = time.perf_counter()
        generation_started = time.perf_counter()
        samples = method.generate(
            case,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=case_seed,
        )
        generation_seconds = time.perf_counter() - generation_started
        total_generation_seconds += generation_seconds
        write_started = time.perf_counter()
        write_generated_posterior(
            output_path,
            samples,
            case=case,
            method=args.method,
            checkpoint=args.checkpoint,
            profile=args.profile,
            seed=case_seed,
        )
        write_seconds = time.perf_counter() - write_started
        visualization_started = time.perf_counter()
        reference = dataset.load_reference(case_path)
        selections = {}
        for field in case.target_fields:
            target_samples = samples[field]
            generated_weights = np.full(
                len(target_samples),
                1.0 / len(target_samples),
                dtype=np.float64,
            )
            figure_name = (
                "posterior_samples.png"
                if len(case.target_fields) == 1
                else f"posterior_samples_{field}.png"
            )
            figure_path = output_path.parent / figure_name
            selections[field] = write_posterior_samples_figure(
                figure_path,
                case=case,
                reference=reference,
                generated=target_samples,
                generated_weights=generated_weights,
                field=field,
                seed=args.visualization_seed,
            )
            print(f"Saved {figure_path}")
        visualization = (
            selections[case.target_field]
            if len(case.target_fields) == 1
            else selections
        )
        write_json(output_path.parent / "visualization.json", visualization)
        print(f"Saved {output_path}")
        visualization_seconds = time.perf_counter() - visualization_started
        case_seconds = time.perf_counter() - case_started
        generated_cases += 1
        runtime_cases[case.case_id] = {
            "skipped": False,
            "seed": case_seed,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "generation_seconds": generation_seconds,
            "write_seconds": write_seconds,
            "visualization_seconds": visualization_seconds,
            "total_seconds": case_seconds,
            "generation_samples_per_second": (
                args.num_samples / generation_seconds
                if generation_seconds > 0
                else None
            ),
        }

    total_seconds = time.perf_counter() - total_started
    generated_samples = generated_cases * args.num_samples
    runtime.update(
        {
            "generated_cases": generated_cases,
            "skipped_cases": skipped_cases,
            "cases": runtime_cases,
            "total_seconds": total_seconds,
            "generation_seconds": total_generation_seconds,
            "generated_samples": generated_samples,
            "wall_samples_per_second": (
                generated_samples / total_seconds if total_seconds > 0 else None
            ),
            "generation_samples_per_second": (
                generated_samples / total_generation_seconds
                if total_generation_seconds > 0
                else None
            ),
        }
    )
    write_json(output_root / "runtime.json", runtime)
    print(f"Saved {output_root / 'runtime.json'}")


if __name__ == "__main__":
    main()
