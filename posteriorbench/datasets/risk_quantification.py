"""Single risk field with an identity forward map and sparse observations."""

import numpy as np

from posteriorbench.datasets.base import (
    BenchmarkCase, DatasetAdapter, HFCaseRef, ReferencePosterior, _as_weights,
)


class RiskQuantificationDataset(DatasetAdapter):
    name = "risk_quantification"
    target_fields = ("risk",)
    observed_field = "risk"

    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        if channels != self.target_fields:
            raise ValueError(f"RiskQuantification expects ('risk',), got {channels}")

    def _load_hf_case(self, ref: HFCaseRef) -> BenchmarkCase:
        # Select metadata before loading: inference never reads reference fields
        # or posterior samples, even though they share the same Arrow row.
        metadata = self._load_hf_dataset(ref.source)[ref.split].select_columns(
            ["metadata"]
        )[ref.index]["metadata"]
        shape = (len(metadata["tau_grid"]), len(metadata["x_grid"]))
        raw_coords = np.asarray(metadata["sensor_coords"])
        coords = raw_coords.astype(np.int64)
        values = np.asarray(metadata["observation_values"], dtype=np.float32)
        if coords.shape != (int(metadata["M"]), 2) or values.shape != (len(coords),):
            raise ValueError("Risk observation coordinates/values do not match M")
        if not np.array_equal(coords, raw_coords):
            raise ValueError("Risk observation coordinates must be integers")
        if (np.any(coords[:, 0] <= 0) or np.any(coords[:, 0] >= shape[0])
                or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= shape[1])
                or len(np.unique(coords, axis=0)) != len(coords)):
            raise ValueError("Risk coordinates must be unique, in bounds, and exclude tau=0")
        if not np.isfinite(values).all():
            raise ValueError("Risk observations must be finite")
        observed = np.zeros(shape, dtype=np.float32)
        observed[coords[:, 0], coords[:, 1]] = values
        return BenchmarkCase(
            case_id=ref.stem, path=f"{ref.source}::{ref.split}[{ref.index}]",
            dataset=self.name, fields={"risk": observed}, sensor_coords=coords,
            target_fields=self.target_fields, observed_field=self.observed_field,
            observation_operator=self.observation_operator,
        )

    def _load_hf_reference(self, ref: HFCaseRef) -> ReferencePosterior:
        row = self._load_hf_row(ref)
        samples = np.asarray(row["posterior_samples_u"], dtype=np.float32)
        if samples.ndim != 3 or not np.isfinite(samples).all():
            raise ValueError("Risk posterior samples must be finite [N,H,W] arrays")
        return ReferencePosterior(
            fields={"risk": samples},
            weights=_as_weights(row["posterior_weights"], len(samples)),
            truth_fields={"risk": np.asarray(row["u_ref"], dtype=np.float32)},
        )
