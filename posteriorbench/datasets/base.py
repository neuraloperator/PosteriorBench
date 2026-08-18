from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    path: Path | str
    dataset: str
    fields: dict[str, np.ndarray]
    sensor_coords: np.ndarray | None
    target_fields: tuple[str, ...]
    observed_field: str
    observation_operator: str

    @property
    def resolution(self) -> tuple[int, int]:
        shape = self.fields[self.target_fields[0]].shape
        return int(shape[-2]), int(shape[-1])

    @property
    def observation_resolution(self) -> tuple[int, int]:
        shape = self.fields[self.observed_field].shape
        return int(shape[-2]), int(shape[-1])

    @property
    def target_field(self) -> str:
        if len(self.target_fields) != 1:
            raise ValueError(
                f"{self.dataset} has multiple target fields: {self.target_fields}"
            )
        return self.target_fields[0]


@dataclass(frozen=True)
class ReferencePosterior:
    fields: dict[str, np.ndarray]
    weights: np.ndarray

    @property
    def samples(self) -> np.ndarray:
        if len(self.fields) != 1:
            raise ValueError(
                f"Reference posterior has multiple fields: {tuple(self.fields)}"
            )
        return next(iter(self.fields.values()))


@dataclass(frozen=True)
class HFCaseRef:
    source: str
    split: str
    index: int
    stem: str


HF_POSTERIOR_DATASET_NAMES = {
    "ccs": "CCS_Multimode",
    "darcy": "Darcy_Multimode",
    "light_transport": "LTMI_Multimode",
    "poisson": "Poisson_Multimode",
}

HF_FIELD_PREFIX = {
    "ccs": "ccs",
    "darcy": "darcy",
    "light_transport": "light_transport",
    "poisson": "poisson",
}

HF_SPLIT_ORDER = ("validation", "test")


def _require_datasets():
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face dataset sources require the 'datasets' package. "
            "Install it with `pip install datasets huggingface_hub pyarrow`."
        ) from exc
    return Dataset, DatasetDict, load_from_disk


def _normalize_hf_source(source: str | Path, dataset_name: str) -> str | None:
    raw = str(source)
    hf_dataset = HF_POSTERIOR_DATASET_NAMES[dataset_name]
    expected_suffix = f"PosteriorDataset_hf/{hf_dataset}"

    if raw.startswith("hf://"):
        path = raw
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
        if (local / "dataset_dict.json").is_file():
            return raw
        if (local / expected_suffix / "dataset_dict.json").is_file():
            return str(local / expected_suffix)
        if (local / "PosteriorDataset_hf" / hf_dataset / "dataset_dict.json").is_file():
            return str(local / "PosteriorDataset_hf" / hf_dataset)
        if (local / hf_dataset / "dataset_dict.json").is_file():
            return str(local / hf_dataset)
        return None

    if path.rstrip("/").endswith(expected_suffix):
        return path.rstrip("/")
    if path.rstrip("/").endswith("PosteriorDataset_hf"):
        return f"{path.rstrip('/')}/{hf_dataset}"
    if path.rstrip("/").endswith(hf_dataset):
        return path.rstrip("/")
    return f"{path.rstrip('/')}/{expected_suffix}"


def _squeeze_single_channel(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W] or [1,H,W], got {array.shape}")
    return array.astype(np.float32, copy=False)


def _split_two_channels(value: Any, names: tuple[str, str]) -> dict[str, np.ndarray]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(
            f"Expected stacked channels {names} with shape [2,H,W], got {array.shape}"
        )
    return {
        names[0]: array[0].astype(np.float32, copy=False),
        names[1]: array[1].astype(np.float32, copy=False),
    }


def _as_weights(value: Any, sample_count: int) -> np.ndarray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape != (sample_count,):
        raise ValueError(
            f"posterior_weights must have shape ({sample_count},), got {weights.shape}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("posterior_weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("posterior_weights must sum to a positive value")
    return weights / total


def _sensor_coords_from_metadata(row: dict[str, Any]) -> np.ndarray | None:
    metadata = row.get("metadata") or {}
    coords = metadata.get("sensor_coords")
    if coords is None:
        return None
    array = np.asarray(coords, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"metadata.sensor_coords must have shape [N,2], got {array.shape}")
    return array


def _fixed_column_coords(shape: tuple[int, int], columns: tuple[int, ...]) -> np.ndarray:
    height, width = shape
    coords = []
    for column in columns:
        if column < 0 or column >= width:
            raise ValueError(f"Fixed column {column} is outside width {width}")
    for row in range(height):
        coords.extend((row, column) for column in columns)
    return np.asarray(coords, dtype=np.int64)


class DatasetAdapter(ABC):
    name: str
    target_fields: tuple[str, ...]
    observed_field: str
    observation_operator = "sparse_points"
    hf_fixed_sensor_columns: tuple[int, ...] = (0, 50)

    @property
    def target_field(self) -> str:
        if len(self.target_fields) != 1:
            raise ValueError(
                f"{self.name} has multiple target fields: {self.target_fields}"
            )
        return self.target_fields[0]

    def discover(self, cases_path: str | Path) -> list[HFCaseRef]:
        hf_source = _normalize_hf_source(cases_path, self.name)
        if hf_source is None:
            raise FileNotFoundError(
                "Cases must be a Hugging Face PosteriorDataset_hf source or "
                f"materialized save_to_disk directory for {self.name}: {cases_path}"
            )
        return self._discover_hf(hf_source)

    def _discover_hf(self, source: str) -> list[HFCaseRef]:
        _, DatasetDict, _ = _require_datasets()
        dataset = self._load_hf_dataset(source)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"HF posterior source must be a DatasetDict: {source}")

        refs: list[HFCaseRef] = []
        prefix = HF_FIELD_PREFIX[self.name]
        offset = 0
        split_names = [split for split in HF_SPLIT_ORDER if split in dataset]
        split_names.extend(
            split for split in dataset.keys() if split not in HF_SPLIT_ORDER
        )
        for split in split_names:
            split_dataset = dataset[split]
            for index in range(len(split_dataset)):
                case_number = offset + index + 1
                refs.append(
                    HFCaseRef(
                        source=source,
                        split=split,
                        index=index,
                        stem=f"{prefix}_{case_number}",
                    )
                )
            offset += len(split_dataset)
        if not refs:
            raise FileNotFoundError(f"No HF cases found in {source}")
        return refs

    def _load_hf_dataset(self, source: str):
        cache = getattr(self, "_hf_dataset_cache", None)
        if cache is None:
            cache = {}
            self._hf_dataset_cache = cache
        if source not in cache:
            _, _, load_from_disk = _require_datasets()
            cache[source] = load_from_disk(source).with_format("numpy")
        return cache[source]

    def _load_hf_row(self, ref: HFCaseRef) -> dict[str, Any]:
        dataset = self._load_hf_dataset(ref.source)
        return dataset[ref.split][ref.index]

    def _hf_case_id(self, ref: HFCaseRef, _row: dict[str, Any]) -> str:
        return ref.stem

    def _load_hf_case(self, ref: HFCaseRef) -> BenchmarkCase:
        row = self._load_hf_row(ref)
        case_id = self._hf_case_id(ref, row)

        if self.name in {"poisson", "darcy"}:
            target = _squeeze_single_channel(row["a_ref"], name="a_ref")
            observed = _squeeze_single_channel(row["u_ref"], name="u_ref")
            sensor_coords = _sensor_coords_from_metadata(row)
            fields = {
                self.target_field: target,
                self.observed_field: observed,
            }
        elif self.name == "light_transport":
            targets = _split_two_channels(row["a_ref"], self.target_fields)
            metadata = row.get("metadata") or {}
            observed_low = metadata.get("obs_u_channel0")
            if observed_low is None:
                raise KeyError(
                    "HF Light Transport posterior rows must provide "
                    "metadata.obs_u_channel0"
                )
            observed = _squeeze_single_channel(observed_low, name="metadata.obs_u_channel0")
            sensor_coords = None
            fields = {
                **targets,
                self.observed_field: observed,
            }
        elif self.name == "ccs":
            target = _squeeze_single_channel(row["a_ref"], name="a_ref")
            observed = _squeeze_single_channel(row["u_ref"], name="u_ref")
            sensor_coords = _sensor_coords_from_metadata(row)
            if sensor_coords is None:
                sensor_coords = _fixed_column_coords(
                    target.shape,
                    self.hf_fixed_sensor_columns,
                )
            fields = {
                self.target_field: target,
                self.observed_field: observed,
            }
        else:
            raise ValueError(f"Unsupported HF dataset adapter: {self.name}")

        return BenchmarkCase(
            case_id=case_id,
            path=f"{ref.source}::{ref.split}[{ref.index}]",
            dataset=self.name,
            fields=fields,
            sensor_coords=sensor_coords,
            target_fields=self.target_fields,
            observed_field=self.observed_field,
            observation_operator=self.observation_operator,
        )

    def load_case(self, path: HFCaseRef) -> BenchmarkCase:
        if not isinstance(path, HFCaseRef):
            raise TypeError(
                "load_case expects an HFCaseRef returned by discover(); "
                f"got {type(path).__name__}"
            )
        return self._load_hf_case(path)

    def _load_hf_reference(self, ref: HFCaseRef) -> ReferencePosterior:
        row = self._load_hf_row(ref)

        if self.name in {"poisson", "darcy", "ccs"}:
            field = self.target_field
            samples = np.asarray(row["posterior_samples_a"], dtype=np.float32)
            if samples.ndim == 4 and samples.shape[1] == 1:
                samples = samples[:, 0]
            fields = {field: samples.astype(np.float32, copy=False)}
        elif self.name == "light_transport":
            samples = np.asarray(row["posterior_samples_a"], dtype=np.float32)
            if samples.ndim != 4 or samples.shape[1] != 2:
                raise ValueError(
                    "HF Light Transport posterior_samples_a must have shape "
                    f"[N,2,H,W], got {samples.shape}"
                )
            fields = {
                self.target_fields[0]: samples[:, 0].astype(np.float32, copy=False),
                self.target_fields[1]: samples[:, 1].astype(np.float32, copy=False),
            }
        else:
            raise ValueError(f"Unsupported HF dataset adapter: {self.name}")

        sample_counts = {len(samples) for samples in fields.values()}
        if len(sample_counts) != 1:
            raise ValueError(f"HF posterior sample count mismatch: {sample_counts}")
        weights = _as_weights(row["posterior_weights"], sample_counts.pop())
        return ReferencePosterior(fields=fields, weights=weights)

    def load_reference(self, path: HFCaseRef) -> ReferencePosterior:
        if not isinstance(path, HFCaseRef):
            raise TypeError(
                "load_reference expects an HFCaseRef returned by discover(); "
                f"got {type(path).__name__}"
            )
        return self._load_hf_reference(path)

    @abstractmethod
    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        raise NotImplementedError
