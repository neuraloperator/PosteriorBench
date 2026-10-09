from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


HF_TRAINING_DATASET_NAMES = {
    "ccs": "CCS_Multimode",
    "darcy": "Darcy_Multimode",
    "light_transport": "LTMI_Multimode",
    "poisson": "Poisson_Multimode",
}

HF_TRAINING_DATASET_TO_NAME = {
    value: key for key, value in HF_TRAINING_DATASET_NAMES.items()
}

HF_TRAINING_CHANNELS = {
    "ccs": ("x", "y"),
    "darcy": ("a", "u"),
    "light_transport": ("sigma_t1", "sigma_t2", "u"),
    "poisson": ("f", "phi"),
}

HF_TRAINING_AVAILABLE_CHANNELS = {
    "ccs": ("x", "y"),
    "darcy": ("a", "u"),
    "light_transport": ("sigma_t1", "sigma_t2", "u", "u2"),
    "poisson": ("f", "phi"),
}


def _require_datasets():
    try:
        from datasets import Dataset, load_from_disk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face training sources require the 'datasets' package. "
            "Install it with `pip install datasets huggingface_hub pyarrow`."
        ) from exc
    return Dataset, load_from_disk


def _normalize_hf_training_source(source: str | Path, dataset_name: str | None = None) -> str | None:
    raw = str(source)
    expected_dataset = (
        HF_TRAINING_DATASET_NAMES.get(dataset_name)
        if dataset_name is not None
        else None
    )

    if raw.startswith("hf://"):
        path = raw.rstrip("/")
    elif raw.startswith("https://huggingface.co/datasets/"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0] != "datasets":
            return None
        repo = "/".join(parts[1:3])
        if "tree" in parts:
            tree_index = parts.index("tree")
            subpath = "/".join(parts[tree_index + 2 :])
        else:
            subpath = ""
        path = f"hf://datasets/{repo}"
        if subpath:
            path = f"{path}/{subpath}"
    else:
        local = Path(raw)
        if (local / "state.json").is_file() and (local / "dataset_info.json").is_file():
            return raw
        if expected_dataset is not None:
            candidates = (
                local / "PDEFieldDataset_hf" / expected_dataset,
                local / expected_dataset,
            )
            for candidate in candidates:
                if (candidate / "state.json").is_file() and (
                    candidate / "dataset_info.json"
                ).is_file():
                    return str(candidate)
        return None

    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "PDEFieldDataset_hf":
        return path
    if parts and parts[-1] in HF_TRAINING_DATASET_TO_NAME:
        return path
    if path.endswith("/PDEFieldDataset_hf"):
        if expected_dataset is None:
            raise ValueError(
                "HF training source points to PDEFieldDataset_hf; pass a dataset-specific "
                "path or provide dataset_name"
            )
        return f"{path}/{expected_dataset}"
    if expected_dataset is not None:
        return f"{path}/PDEFieldDataset_hf/{expected_dataset}"
    return None


def _stats_dict(names: tuple[str, ...], values: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(values["mean"][index]),
            "std": float(values["std"][index]),
            "min": float(values["min"][index]),
            "max": float(values["max"][index]),
        }
        for index, name in enumerate(names)
    }


def _metadata_from_hf(source: str, dataset: Any) -> dict[str, Any]:
    meta_path = Path(source) / "metadata.json"
    metadata: dict[str, Any] | None = None
    if not source.startswith("hf://") and meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = _remote_hf_metadata(source)

    if metadata is None:
        # Remote metadata fallback for the anonymousmay/PosteriorBench reference.
        # Values mirror the downloaded metadata.json files.
        name_from_path = next(
            (
                HF_TRAINING_DATASET_TO_NAME[part]
                for part in source.split("/")
                if part in HF_TRAINING_DATASET_TO_NAME
            ),
            None,
        )
        if name_from_path is None:
            raise ValueError(f"Cannot infer HF training dataset from {source}")
        metadata = _default_hf_metadata(name_from_path)

    hf_name = str(metadata.get("dataset_name") or metadata.get("name") or "")
    dataset_name = HF_TRAINING_DATASET_TO_NAME.get(hf_name)
    if dataset_name is None and hf_name == "darcy":
        dataset_name = "darcy"
    if dataset_name is None:
        raise ValueError(f"Unsupported HF training dataset metadata name: {hf_name!r}")

    shape_value = metadata["shape"]
    if dataset_name == "darcy" and len(shape_value) == 3:
        resolution = [int(shape_value[1]), int(shape_value[2])]
    else:
        resolution = [int(shape_value[0]), int(shape_value[1])]

    if dataset_name == "poisson":
        stats = _stats_dict(("f", "phi"), metadata["stats"])
        for old, new in (("alpha", "alpha"), ("tau", "tau")):
            if old in metadata.get("params_stats", {}):
                stats[new] = {
                    key: float(value)
                    for key, value in metadata["params_stats"][old].items()
                }
    elif dataset_name == "darcy":
        stats = _stats_dict(("a", "u"), metadata["stats"])
    elif dataset_name == "light_transport":
        stats = _stats_dict(("sigma_t1", "sigma_t2", "u", "u2"), metadata["stats"])
    elif dataset_name == "ccs":
        stats = _stats_dict(("x", "y"), metadata["stats"])
    else:
        raise AssertionError("validated dataset_name became invalid")

    return {
        "schema_version": "posteriorbench.training.v1",
        "dataset": dataset_name,
        "channels": list(HF_TRAINING_CHANNELS[dataset_name]),
        "auxiliary_fields": [],
        "num_samples": len(dataset),
        "resolution": resolution,
        "dtypes": {
            channel: "float32"
            for channel in HF_TRAINING_AVAILABLE_CHANNELS[dataset_name]
        },
        "source": {
            "type": "huggingface",
            "path": source,
            "storage_normalization_scale": 0.5 if dataset_name == "darcy" else None,
        },
        "statistics": stats,
    }


def _remote_hf_metadata(source: str) -> dict[str, Any] | None:
    if not source.startswith("hf://datasets/"):
        return None
    parts = source[len("hf://datasets/") :].strip("/").split("/")
    if len(parts) < 3:
        return None
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:])
    filename = f"{subpath}/metadata.json"
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
    except Exception:
        return None
    with Path(path).open() as handle:
        return json.load(handle)


def _default_hf_metadata(dataset_name: str) -> dict[str, Any]:
    defaults = {
        "poisson": {
            "dataset_name": "Poisson_Multimode",
            "shape": [128, 128],
            "stats": {
                "mean": [-5.112124255024497e-12, 6.660705921767042e-06],
                "std": [0.2719037491997424, 0.0038558472542799586],
                "min": [-3.0957822799682617, -0.029318762943148613],
                "max": [2.634230613708496, 0.031122952699661255],
            },
            "params_stats": {
                "alpha": {
                    "mean": 2.2492973804473877,
                    "std": 0.4321150779724121,
                    "min": 1.5000313520431519,
                    "max": 2.9999794960021973,
                },
                "tau": {
                    "mean": 3.001169443130493,
                    "std": 0.5747162103652954,
                    "min": 2.0000288486480713,
                    "max": 3.9999659061431885,
                },
            },
        },
        "darcy": {
            "name": "darcy",
            "shape": [2, 128, 128],
            "stats": {
                "mean": [7.5, 0.00569201936],
                "std": [4.5, 0.00379030361],
                "min": [3.0, -0.28737752],
                "max": [12.0, 0.11770357],
            },
        },
        "light_transport": {
            "dataset_name": "LTMI_Multimode",
            "shape": [64, 64],
            "stats": {
                "mean": [
                    2.516578340749906,
                    2.518172909586568,
                    0.3230067392023618,
                    0.46580631291939256,
                ],
                "std": [
                    1.7554190007156716,
                    1.7551658526673308,
                    0.1308156101844342,
                    0.45600807851895414,
                ],
                "min": [0.009999999776482582, 0.009999999776482582, 0.0, 0.0],
                "max": [5.0, 5.0, 0.6829288005828857, 2.611924886703491],
            },
        },
        "ccs": {
            "dataset_name": "CCS_Multimode",
            "shape": [64, 200],
            "stats": {
                "mean": [248.86026000976562, 0.08489078283309937],
                "std": [212.1194610595703, 0.21714571118354797],
                "min": [0.09999999403953552, 0.0],
                "max": [2335.570556640625, 1.0000230073928833],
            },
        },
    }
    return defaults[dataset_name]


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != "posteriorbench.training.v1":
        raise ValueError(
            "Expected posteriorbench.training.v1 metadata, got "
            f"{metadata.get('schema_version')!r}"
        )
    channels = tuple(metadata["channels"])
    if not channels:
        raise ValueError("Training metadata must define at least one model channel")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Training channels must be unique, got {channels}")


def _as_indices(indices: Any) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"indices must be a 1D array, got {array.shape}")
    return array


class TrainingFieldSource:
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.metadata["dataset"])

    @property
    def num_samples(self) -> int:
        return int(self.metadata["num_samples"])

    @property
    def resolution(self) -> tuple[int, int]:
        height, width = self.metadata["resolution"]
        return int(height), int(width)

    def channel_shapes(self, channels: tuple[str, ...]) -> list[tuple[int, int, int]]:
        for channel in channels:
            if channel not in self.metadata["statistics"]:
                raise KeyError(f"Unknown training channel '{channel}'")
        height, width = self.resolution
        return [(self.num_samples, height, width) for _ in channels]

    def read_channels(self, indices: Any, channels: tuple[str, ...]) -> np.ndarray:
        raise NotImplementedError


class HFTrainingFieldSource(TrainingFieldSource):
    def __init__(self, source: str, dataset_name: str | None = None):
        self.source = _normalize_hf_training_source(source, dataset_name)
        if self.source is None:
            raise ValueError(f"Not a Hugging Face training source: {source}")
        Dataset, load_from_disk = _require_datasets()
        dataset = load_from_disk(self.source).with_format("numpy")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"HF training source must be a Dataset: {self.source}")
        self._dataset = dataset
        self.metadata = _metadata_from_hf(self.source, dataset)
        _validate_metadata(self.metadata)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def _load_dataset(self):
        if self._dataset is None:
            _, load_from_disk = _require_datasets()
            self._dataset = load_from_disk(self.source).with_format("numpy")
        return self._dataset

    def _batch(self, indices: np.ndarray) -> dict[str, Any]:
        dataset = self._load_dataset()
        return dataset[indices.tolist()]

    def read_channels(self, indices: Any, channels: tuple[str, ...]) -> np.ndarray:
        index_array = _as_indices(indices)
        batch = self._batch(index_array)
        arrays = self._all_channels_from_batch(batch)
        missing = [channel for channel in channels if channel not in arrays]
        if missing:
            raise KeyError(
                f"Missing HF training channels {missing}; available: {sorted(arrays)}"
            )
        return np.stack([arrays[channel] for channel in channels], axis=1).astype(np.float32)

    def _all_channels_from_batch(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        dataset = self.name
        if dataset == "poisson":
            return {
                "f": np.asarray(batch["a"], dtype=np.float32),
                "phi": np.asarray(batch["u"], dtype=np.float32),
            }
        if dataset == "darcy":
            data = np.asarray(batch["data"], dtype=np.float32)
            if data.ndim != 4 or data.shape[1] != 2:
                raise ValueError(f"Darcy HF data must have shape [N,2,H,W], got {data.shape}")
            return {
                "a": self._denormalize_hf_channel("a", data[:, 0]),
                "u": self._denormalize_hf_channel("u", data[:, 1]),
            }
        if dataset == "light_transport":
            a = np.asarray(batch["a"], dtype=np.float32)
            u = np.asarray(batch["u"], dtype=np.float32)
            if a.ndim != 4 or a.shape[1] != 2:
                raise ValueError(f"LTMI HF a must have shape [N,2,H,W], got {a.shape}")
            if u.ndim != 4 or u.shape[1] != 2:
                raise ValueError(f"LTMI HF u must have shape [N,2,H,W], got {u.shape}")
            return {
                "sigma_t1": a[:, 0],
                "sigma_t2": a[:, 1],
                "u": u[:, 0],
                "u2": u[:, 1],
            }
        if dataset == "ccs":
            a = np.asarray(batch["a"], dtype=np.float32)
            u = np.asarray(batch["u"], dtype=np.float32)
            if a.ndim != 4 or a.shape[1] != 1:
                raise ValueError(f"CCS HF a must have shape [N,1,H,W], got {a.shape}")
            if u.ndim != 4 or u.shape[1] != 1:
                raise ValueError(f"CCS HF u must have shape [N,1,H,W], got {u.shape}")
            return {
                "x": a[:, 0],
                "y": u[:, 0],
            }
        raise AssertionError("validated dataset became invalid")

    def _denormalize_hf_channel(self, channel: str, values: np.ndarray) -> np.ndarray:
        scale = self.metadata["source"].get("storage_normalization_scale")
        if scale is None:
            return values.astype(np.float32, copy=False)
        stats = self.metadata["statistics"][channel]
        return (
            values.astype(np.float32) * (float(stats["std"]) / float(scale))
            + float(stats["mean"])
        ).astype(np.float32, copy=False)


def open_training_field_source(
    path: str | Path,
    *,
    dataset_name: str | None = None,
) -> TrainingFieldSource:
    hf_source = _normalize_hf_training_source(path, dataset_name)
    if hf_source is None:
        raise ValueError(
            "Training data must be a Hugging Face PDEFieldDataset_hf source "
            f"or materialized save_to_disk directory, got: {path}"
        )
    return HFTrainingFieldSource(hf_source, dataset_name)


def is_training_field_source(
    path: str | Path,
    *,
    dataset_name: str | None = None,
) -> bool:
    return _normalize_hf_training_source(path, dataset_name) is not None
