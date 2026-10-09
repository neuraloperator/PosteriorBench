from __future__ import annotations

import numpy as np
import torch

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords, spatial_to_config
from posteriorbench.training_source import TrainingFieldSource, open_training_field_source


def _validate_training_channels(
    source: TrainingFieldSource,
    channels: tuple[str, ...],
) -> tuple[int, int, int]:
    shapes = source.channel_shapes(channels)
    if any(len(shape) != 3 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError(f"Expected matching [N,H,W] channels, got {shapes}")
    num_samples, height, width = shapes[0]
    return int(num_samples), int(height), int(width)


def _read_prior_ensemble(
    source: TrainingFieldSource,
    channels: tuple[str, ...],
    indices: np.ndarray,
    resolution: object,
) -> np.ndarray:
    order = np.argsort(indices)
    sorted_indices = np.asarray(indices[order], dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    fields = source.read_channels(sorted_indices, channels).astype(np.float32)
    fields = fields[inverse]
    raw_shape = tuple(int(value) for value in fields.shape[-2:])
    target_shape = as_spatial_shape(resolution)
    if raw_shape == target_shape:
        return fields
    if raw_shape[0] % target_shape[0] != 0 or raw_shape[1] % target_shape[1] != 0:
        raise ValueError(
            f"Prior resolution {raw_shape} must be divisible by requested "
            f"ES-MDA resolution {target_shape}"
        )
    stride_h = raw_shape[0] // target_shape[0]
    stride_w = raw_shape[1] // target_shape[1]
    return fields[:, :, ::stride_h, ::stride_w].astype(np.float32, copy=False)


def _area_average_batch(fields: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if fields.ndim != 3:
        raise ValueError(f"Expected [N,H,W] fields, got {fields.shape}")
    out_h, out_w = output_shape
    in_h, in_w = fields.shape[-2:]
    if in_h % out_h != 0 or in_w % out_w != 0:
        raise ValueError(
            f"Cannot area-average {in_h}x{in_w} fields to {out_h}x{out_w}"
        )
    block_h = in_h // out_h
    block_w = in_w // out_w
    return fields.reshape(-1, out_h, block_h, out_w, block_w).mean(axis=(2, 4))


class ESMDAAdapter(MethodAdapter):
    """Augmented-state ES-MDA baseline over paired HF training fields.

    The checkpoint argument is a materialized ``PDEFieldDataset_hf`` source.
    ES-MDA updates the joint state containing target fields and the observed
    physical field, then writes only the benchmark target fields.
    """

    def load(self) -> None:
        if self.profile.method != "esmda":
            raise ValueError(
                f"ES-MDA adapter cannot use method profile '{self.profile.method}'"
            )
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match "
                f"'{self.dataset.name}'"
            )
        self.dataset.validate_model_channels(self.profile.channels)
        if self.dataset.observed_field not in self.profile.channels:
            raise ValueError(
                f"ES-MDA channels {self.profile.channels} do not contain observed "
                f"field '{self.dataset.observed_field}'"
            )
        for field in self.dataset.target_fields:
            if field not in self.profile.channels:
                raise ValueError(
                    f"ES-MDA channels {self.profile.channels} do not contain target "
                    f"field '{field}'"
                )
        if bool(self.profile.guidance.get("pending", False)) and not bool(
            self.profile.guidance.get("allow_pending", False)
        ):
            raise ValueError(
                f"ES-MDA profile for dataset '{self.dataset.name}' is still pending; "
                "run only --dry-run until observation-error settings are finalized"
            )

        self.prior_source = open_training_field_source(
            self.checkpoint,
            dataset_name=self.dataset.name,
        )
        self.num_prior_samples, raw_height, raw_width = _validate_training_channels(
            self.prior_source,
            self.profile.channels,
        )
        self.raw_shape = (raw_height, raw_width)
        metadata = self.prior_source.metadata
        if str(metadata["dataset"]) != self.dataset.name:
            raise ValueError(
                f"Prior ensemble dataset '{metadata['dataset']}' does not match "
                f"'{self.dataset.name}'"
            )
        self.model_shape = as_spatial_shape(self.profile.resolution or self.raw_shape)
        self.model_resolution = spatial_to_config(self.model_shape)
        if (
            self.raw_shape[0] % self.model_shape[0] != 0
            or self.raw_shape[1] % self.model_shape[1] != 0
        ):
            raise ValueError(
                f"Prior resolution {self.raw_shape} must be divisible by "
                f"profile resolution {self.model_shape}"
            )

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        self.sampling = dict(self.profile.sampling)
        self.guidance = dict(self.profile.guidance)

    def _alphas(self) -> list[float]:
        explicit = self.sampling.get("inflation_factors")
        if explicit is not None:
            alphas = [float(value) for value in explicit]
        else:
            num_assimilations = int(self.sampling.get("num_assimilations", 4))
            if num_assimilations <= 0:
                raise ValueError("num_assimilations must be positive")
            scheme = str(self.sampling.get("inflation", "constant")).lower()
            if scheme == "constant":
                alphas = [float(num_assimilations)] * num_assimilations
            elif scheme == "geometric":
                gamma = float(self.sampling.get("geometric_ratio", 0.7))
                if not 0.0 < gamma <= 1.0:
                    raise ValueError("geometric_ratio must be in (0, 1]")
                alpha1 = sum(gamma ** (-k) for k in range(num_assimilations))
                alphas = [alpha1 * gamma**k for k in range(num_assimilations)]
            else:
                raise ValueError(f"Unsupported ES-MDA inflation scheme: {scheme}")
        if any(alpha <= 0 or not np.isfinite(alpha) for alpha in alphas):
            raise ValueError(f"Invalid ES-MDA inflation factors: {alphas}")
        inverse_sum = sum(1.0 / alpha for alpha in alphas)
        if not np.isclose(inverse_sum, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "ES-MDA inflation factors must satisfy sum(1/alpha_k)=1, got "
                f"{inverse_sum}"
            )
        return alphas

    def _ensemble_indices(self, ensemble_size: int, rng: np.random.Generator) -> np.ndarray:
        if ensemble_size <= 1:
            raise ValueError("ES-MDA ensemble_size must be greater than one")
        replace = ensemble_size > self.num_prior_samples
        return rng.choice(self.num_prior_samples, size=ensemble_size, replace=replace)

    def _observation_matrix(self, state: np.ndarray, case: BenchmarkCase) -> np.ndarray:
        observed_channel = self.profile.channels.index(case.observed_field)
        observed = state[:, observed_channel]
        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = scale_sensor_coords(
                case.sensor_coords,
                case.observation_resolution,
                self.model_shape,
            )
            return observed[:, coords[:, 0], coords[:, 1]].astype(np.float64)
        if case.observation_operator == "area_average":
            pooled = _area_average_batch(observed, case.observation_resolution)
            return pooled.reshape(len(state), -1).astype(np.float64)
        raise ValueError(f"Unsupported observation operator: {case.observation_operator}")

    def _observed_vector(self, case: BenchmarkCase) -> np.ndarray:
        observed = case.fields[case.observed_field]
        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = case.sensor_coords
            return observed[coords[:, 0], coords[:, 1]].astype(np.float64)
        if case.observation_operator == "area_average":
            return observed.reshape(-1).astype(np.float64)
        raise ValueError(f"Unsupported observation operator: {case.observation_operator}")

    def _observation_std(self, predictions: np.ndarray) -> np.ndarray:
        config = dict(self.guidance.get("observation_error", {}))
        explicit = config.get("std")
        if explicit is not None:
            std = np.asarray(explicit, dtype=np.float64)
            if std.ndim == 0:
                std = np.full(predictions.shape[1], float(std), dtype=np.float64)
            if std.shape != (predictions.shape[1],):
                raise ValueError(
                    f"observation_error.std must be scalar or shape "
                    f"({predictions.shape[1]},), got {std.shape}"
                )
        else:
            mode = str(config.get("mode", "prior_std_fraction")).lower()
            if mode != "prior_std_fraction":
                raise ValueError(f"Unsupported observation_error mode: {mode}")
            scale = float(config.get("scale", 0.05))
            std = scale * np.std(predictions, axis=0, ddof=1)
        min_std = float(config.get("min_std", 1.0e-6))
        if min_std <= 0 or not np.isfinite(min_std):
            raise ValueError("observation_error.min_std must be finite and positive")
        std = np.maximum(std, min_std)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Observation error standard deviations must be positive")
        return std

    def _assimilate(
        self,
        state: np.ndarray,
        observed_vector: np.ndarray,
        case: BenchmarkCase,
        rng: np.random.Generator,
    ) -> np.ndarray:
        alphas = self._alphas()
        state_2d = state.reshape(state.shape[0], -1).astype(np.float64)
        predictions = self._observation_matrix(state, case)
        obs_std = self._observation_std(predictions)
        obs_var = obs_std**2
        jitter = float(
            dict(self.guidance.get("observation_error", {})).get("jitter", 1.0e-8)
        )

        for alpha in alphas:
            predictions = self._observation_matrix(
                state_2d.reshape(state.shape),
                case,
            )
            state_anom = (state_2d - state_2d.mean(axis=0, keepdims=True)).T
            pred_anom = (predictions - predictions.mean(axis=0, keepdims=True)).T
            scale = np.sqrt(state.shape[0] - 1)
            state_anom /= scale
            pred_anom /= scale

            c_zy = state_anom @ pred_anom.T
            c_yy = pred_anom @ pred_anom.T
            system = c_yy + np.diag(alpha * obs_var)
            diag_scale = max(float(np.mean(np.diag(system))), 1.0)
            system[np.diag_indices_from(system)] += jitter * diag_scale

            perturb = rng.normal(
                loc=0.0,
                scale=np.sqrt(alpha) * obs_std,
                size=predictions.shape,
            )
            innovation = observed_vector[None, :] + perturb - predictions
            weights = np.linalg.solve(system, innovation.T)
            state_2d += (c_zy @ weights).T

        return state_2d.reshape(state.shape).astype(np.float32)

    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        rng = np.random.default_rng(seed)
        ensemble_size = max(
            int(self.sampling.get("ensemble_size", 512)),
            int(num_samples),
        )
        indices = self._ensemble_indices(ensemble_size, rng)
        state = _read_prior_ensemble(
            self.prior_source,
            self.profile.channels,
            indices,
            self.model_shape,
        )
        observed_vector = self._observed_vector(case)
        updated = self._assimilate(state, observed_vector, case, rng)

        tensor = torch.from_numpy(updated[:num_samples])
        tensor = self.normalizer.apply_postprocess(tensor).numpy()
        return {
            field: tensor[:, self.profile.channels.index(field)].astype(
                np.float32,
                copy=False,
            )
            for field in case.target_fields
        }
