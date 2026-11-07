"""Export u-only HF data from complete pools; never resample duplicate fields."""

from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import os
import numpy as np
import pipeline as pp
import safe_explorer_core as prior
from dataset_io import (
    config_dict,
    config_from_dict,
    parameter_arrays,
    sha256,
    save_parameters,
    environment_info,
)


def select_indices(ids, distances, count, seed, case_index):
    ids = np.asarray(ids)
    distances = np.asarray(distances)
    order = np.argsort(ids)
    ids = ids[order]
    distances = distances[order]
    if len(np.unique(ids)) != len(ids) or len(ids) < count:
        raise ValueError("Insufficient unique accepted candidates")
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_index]))
    pick = rng.choice(len(ids), count, replace=False)
    pick = pick[np.lexsort((ids[pick], distances[pick]))]
    return ids[pick], distances[pick]


def checked_manifest(root, cfg, count):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if (
        not m["complete"]
        or m["field_count"] != count
        or m["solver_config"] != config_dict(cfg)
    ):
        raise ValueError("Pool contract mismatch")
    if m["numerical_source_sha256"] != pp.source_hash():
        raise ValueError("Pool numerical source differs")
    stop = 0
    for r in m["blocks"]:
        if r["start"] != stop or r["stop"] <= stop:
            raise ValueError("Noncontiguous pool")
        if sha256(root / r["file"]) != r["sha256"]:
            raise ValueError("Pool checksum mismatch")
        stop = r["stop"]
    if stop != count:
        raise ValueError("Pool size mismatch")
    return m


def train_rows(root, blocks):
    for record in blocks:
        with np.load(Path(root) / record["file"], allow_pickle=False) as z:
            for sid, u in zip(z["sample_ids"], z["u"]):
                yield dict(
                    u=u,
                    metadata=dict(
                        field_id=f"field_{int(sid):06d}", quantity="risk_probability"
                    ),
                )


def posterior_rows(rows):
    yield from rows


def _save(ds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            "Export destination already exists; use an empty destination"
        )
    ds.save_to_disk(str(path), max_shard_size="256MB")


def export_dataset(c, training, candidates, references, out):
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Array2D,
        Array3D,
        Sequence,
        Value,
    )

    training, candidates, references, out = map(
        Path, (training, candidates, references, out)
    )
    cfg = config_from_dict(c["solver"])
    tcfg = replace(cfg, stream=c["training_stream"])
    rcfg = replace(cfg, stream=c["reference_stream"])
    if out.exists():
        if (out / "manifest.json").exists() and json.loads(
            (out / "manifest.json").read_text()
        ).get("complete"):
            old = json.loads((out / "manifest.json").read_text())
            if (
                old.get("configuration") != c
                or old.get("numerical_source_sha256") != pp.source_hash()
            ):
                raise ValueError("Existing export contract differs")
            for file, h in json.loads((out / "checksums.json").read_text()).items():
                if sha256(out / file) != h:
                    raise ValueError("Existing export checksum mismatch")
            return
        if any(out.iterdir()):
            raise FileExistsError(
                "Nonempty incomplete destination; inspect it or use a new destination"
            )
    tm = checked_manifest(training, tcfg, c["training_fields"])
    cm = checked_manifest(candidates, cfg, c["candidate_fields"])
    refmanifest = json.loads((references / "manifest.json").read_text())
    if refmanifest["contract"] != dict(
        configuration=c, numerical_source_sha256=pp.source_hash()
    ) or refmanifest["sha256"] != sha256(references / "references.npz"):
        raise ValueError("Reference contract mismatch")
    with np.load(references / "references.npz", allow_pickle=False) as z:
        truth = z["u"]
        masks = z["masks"]
        rp = parameter_arrays(z)
    np.testing.assert_array_equal(rp["sample_id"], c["reference_ids"])
    n = len(truth)
    keep = c["samples_per_case"]
    shape = (cfg.nt, cfg.nx)
    accepted = [[] for _ in range(n)]
    errors = [[] for _ in range(n)]
    print("Matching complete candidate pool against frozen observations.", flush=True)
    for record in cm["blocks"]:
        with np.load(candidates / record["file"], allow_pickle=False) as z:
            ids = z["sample_ids"]
            fields = z["u"]
        dd = pp.distances(fields, truth, masks)
        for i in range(n):
            mask = dd[:, i] <= c["epsilon"]
            accepted[i].append(ids[mask])
            errors[i].append(dd[mask, i])
    selected = []
    distances = []
    locations = {}
    accepted_counts = []
    for i in range(n):
        ids = np.concatenate(accepted[i])
        dd = np.concatenate(errors[i])
        accepted_counts.append(len(ids))
        if len(ids) < keep:
            raise ValueError(
                f"case_{i:04d}: only {len(ids)} accepted samples, need {keep}; no case replacement or padding"
            )
        chosen, d = select_indices(ids, dd, keep, c["retention_seed"], i)
        selected.append(chosen)
        distances.append(d)
        for j, sid in enumerate(chosen):
            locations.setdefault(int(sid), []).append((i, j))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Complete in a separate staging directory, then publish by atomic rename.
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="export-") as temp:
        dest = Path(temp) / "benchmark"
        dest.mkdir()
        chosen_fields = np.lib.format.open_memmap(
            Path(temp) / "selected.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, keep, *shape),
        )
        chosen_params = [
            {k: np.empty((keep, *v.shape[1:]), v.dtype) for k, v in rp.items()}
            for _ in range(n)
        ]
        seen = set()
        for record in cm["blocks"]:
            wanted = [
                sid for sid in locations if record["start"] <= sid < record["stop"]
            ]
            if not wanted:
                continue
            with np.load(candidates / record["file"], allow_pickle=False) as z:
                p = parameter_arrays(z)
                fields = z["u"]
                ids = z["sample_ids"]
                for sid in wanted:
                    ix = int(np.flatnonzero(ids == sid)[0])
                    for ci, slot in locations[sid]:
                        chosen_fields[ci, slot] = fields[ix]
                        for k in p:
                            chosen_params[ci][k][slot] = p[k][ix]
                    seen.add(sid)
        if seen != set(locations):
            raise ValueError("Selected fields missing")
        feature = Features(
            dict(
                u=Array2D(shape, "float32"),
                metadata=dict(field_id=Value("string"), quantity=Value("string")),
            )
        )
        ds = Dataset.from_generator(
            train_rows,
            gen_kwargs=dict(root=str(training), blocks=tm["blocks"]),
            features=feature,
            cache_dir=str(Path(temp) / "hf_cache"),
            writer_batch_size=32,
        )
        _save(ds, dest / "PDEFieldDataset_hf/RiskQuantification")
        training_records = []
        total = total2 = 0.0
        for r in tm["blocks"]:
            with np.load(training / r["file"], allow_pickle=False) as z:
                p = parameter_arrays(z)
                u = z["u"]
            file = f"parameters/training/{Path(r['file']).name}"
            save_parameters(dest / file, p, tcfg, "exact_parameters_passed_to_solver")
            training_records.append(
                dict(
                    file=file,
                    start=r["start"],
                    stop=r["stop"],
                    count=len(u),
                    sha256=sha256(dest / file),
                )
            )
            total += u.sum(dtype=np.float64)
            total2 += np.square(u.astype(float)).sum()
        mean = float(total / (len(ds) * cfg.nt * cfg.nx))
        std = float(np.sqrt(max(total2 / (len(ds) * cfg.nt * cfg.nx) - mean * mean, 0)))
        pp.atomic_json(
            dest / "PDEFieldDataset_hf/RiskQuantification/metadata.json",
            dict(
                field_names=["u"],
                field_channels={"u": ["risk"]},
                field_slices={"u": [0, 1]},
                statistics={"u": dict(mean=mean, std=std)},
                shape=list(shape),
                num_samples=len(ds),
            ),
        )
        rows = {"validation": [], "test": []}
        case_index = []
        for i in range(n):
            cid = f"case_{i:04d}"
            design = prior.observation_design(
                cfg, c["K_values"][i % len(c["K_values"])], 256, c["observation_seed"]
            )
            np.testing.assert_array_equal(masks[i], design["mask"])
            u = np.asarray(chosen_fields[i])
            d = distances[i]
            logw = -0.5 * (d / c["sigma"]) ** 2
            w = np.exp(logw - logw.max())
            w /= w.sum()
            expected = np.sqrt(
                np.mean(
                    (u[:, masks[i]].astype(float) - truth[i, masks[i]]) ** 2, axis=1
                )
            )
            np.testing.assert_array_equal(d, expected)
            file = f"parameters/cases/{cid}.npz"
            pp.atomic_npz(
                dest / file,
                **{"reference__" + k: v[i : i + 1] for k, v in rp.items()},
                **{"posterior__" + k: v for k, v in chosen_params[i].items()},
            )
            split = "validation" if i < c["validation_cases"] else "test"
            meta = dict(
                case_id=cid,
                sensor_coords=design["sensor_coords"].tolist(),
                observation_values=truth[i, masks[i]].tolist(),
                prefix_lengths=design["lengths"].tolist(),
                initial_state_columns=design["columns"].tolist(),
                x_grid=np.linspace(*cfg.domain, cfg.nx).tolist(),
                tau_grid=np.linspace(0, 1, cfg.nt).tolist(),
                K=int(design["K"]),
                M=256,
                epsilon=c["epsilon"],
                rmse_sigma=c["sigma"],
                quantity="risk_probability",
                field_relation="u is the risk probability field; observations are masked values of u",
                time_axis="remaining_time_to_deadline_1",
                observation_rmse=d.tolist(),
                maximum_observed_pixel_error=np.abs(
                    u[:, masks[i]].astype(float) - truth[i, masks[i]]
                )
                .max(axis=1)
                .tolist(),
                reference_sample_id=int(rp["sample_id"][i]),
                reference_stream=rcfg.stream,
                posterior_sample_ids=selected[i].tolist(),
                posterior_stream=cfg.stream,
            )
            case_index.append(
                dict(
                    case_id=cid,
                    split=split,
                    row=len(rows[split]),
                    reference_id=int(rp["sample_id"][i]),
                    reference_stream=rcfg.stream,
                    posterior_stream=cfg.stream,
                    posterior_sample_ids=selected[i].tolist(),
                    parameters=file,
                    parameter_sha256=sha256(dest / file),
                    accepted_count=accepted_counts[i],
                )
            )
            rows[split].append(
                dict(
                    u_ref=truth[i],
                    posterior_samples_u=u,
                    posterior_weights=w,
                    metadata=meta,
                )
            )
        # Explicit field dtypes; metadata is inferred from the numeric values above.
        inferred = Dataset.from_list([rows["validation"][0]]).features
        inferred["u_ref"] = Array2D(shape, "float32")
        inferred["posterior_samples_u"] = Array3D((keep, *shape), "float32")
        inferred["posterior_weights"] = Sequence(Value("float64"), length=keep)
        post = DatasetDict(
            {
                s: Dataset.from_generator(
                    posterior_rows,
                    gen_kwargs={"rows": v},
                    features=inferred,
                    cache_dir=str(Path(temp) / ("hf_" + s)),
                    writer_batch_size=1,
                )
                for s, v in rows.items()
            }
        )
        _save(post, dest / "PosteriorDataset_hf/RiskQuantification")
        pp.atomic_json(
            dest / "PosteriorDataset_hf/RiskQuantification/metadata.json",
            dict(
                field_names=["u"],
                target_fields=["u"],
                observed_field="u",
                case_count={k: len(v) for k, v in rows.items()},
                posterior_num_samples=keep,
            ),
        )
        manifest = dict(
            format_version=2,
            complete=True,
            kind="benchmark_release",
            field_name="u",
            prior="moving_mix",
            training_fields=len(ds),
            case_index=case_index,
            training_parameters=training_records,
            candidate_config=config_dict(cfg),
            training_config=config_dict(tcfg),
            reference_config=config_dict(rcfg),
            configuration=c,
            numerical_source_sha256=pp.source_hash(),
            parameter_origin="exact_parameters_passed_to_solver",
            environment=environment_info(),
            official_posteriorbench_registration_required=True,
        )
        pp.atomic_json(dest / "manifest.json", manifest)
        pp.atomic_json(
            dest / "checksums.json",
            {
                str(p.relative_to(dest)): sha256(p)
                for p in sorted(dest.rglob("*"))
                if p.is_file()
            },
        )
        os.replace(dest, out)
    print(f"Exported {out}", flush=True)
