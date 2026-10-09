import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training_fundps.noise_samplers import RBFKernel
from utils.yaml_config import Config

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.compat import (
    install_legacy_checkpoint_aliases,
    load_pickle_with_torch_storage_map,
)
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords, spatial_to_config


def _mse_loss(x: torch.Tensor, n_obs) -> torch.Tensor:
    return torch.sum(x**2, dim=(-2, -1)) / n_obs


def _l1_loss(x: torch.Tensor, n_obs) -> torch.Tensor:
    return torch.sum(torch.abs(x), dim=(-2, -1)) / n_obs


def _l2_loss(x: torch.Tensor, n_obs) -> torch.Tensor:
    return torch.sqrt(torch.sum(x**2, dim=(-2, -1)) / n_obs)


def _batched_loss(x: torch.Tensor, n_obs) -> torch.Tensor:
    assert len(x.shape) == 4, "Input tensor must have shape [N,C,H,W]"
    batch_size = x.shape[0]
    channel_losses = torch.sqrt(
        torch.sum(x**2, dim=(0, 2, 3)) / (batch_size * n_obs)
    )
    return channel_losses.repeat(batch_size, 1)


def _huber_loss(x: torch.Tensor, n_obs, delta: float = 1.0) -> torch.Tensor:
    abs_x = torch.abs(x)
    loss = torch.where(
        abs_x < delta,
        0.5 * abs_x**2,
        delta * (abs_x - 0.5 * delta),
    )
    return torch.sum(loss, dim=(-2, -1)) / n_obs


def _get_loss_func(loss_type: str):
    if loss_type == "mse":
        return _mse_loss
    if loss_type == "l1":
        return _l1_loss
    if loss_type == "l2":
        return _l2_loss
    if loss_type == "batched":
        return _batched_loss
    if loss_type.startswith("huber"):
        delta = float(loss_type.split("-")[1]) if "-" in loss_type else 1.0
        return lambda x, n_obs: _huber_loss(x, n_obs, delta=delta)
    raise ValueError(f"Invalid loss type: {loss_type}")


def _get_darcy_residual(x_pred: torch.Tensor) -> torch.Tensor:
    device = x_pred.device
    a_pred = x_pred[:, 0:1]
    u_pred = x_pred[:, 1:2]

    length = a_pred.shape[-1]
    dx = 1 / (length - 1)
    deriv_x = (
        torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device)
        .view(1, 1, 1, 3)
        / (2 * dx)
    )
    deriv_y = (
        torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device)
        .view(1, 1, 3, 1)
        / (2 * dx)
    )

    grad_x = F.conv2d(u_pred, deriv_x, padding=(0, 1))
    grad_y = F.conv2d(u_pred, deriv_y, padding=(1, 0))
    grad_x = a_pred * grad_x
    grad_y = a_pred * grad_y

    div_x = F.conv2d(grad_x, deriv_x, padding=(0, 1))
    div_y = F.conv2d(grad_y, deriv_y, padding=(1, 0))
    return (div_x + div_y + 1)[..., 2:-2, 2:-2]


def _get_poisson_residual(x_pred: torch.Tensor) -> torch.Tensor:
    a_pred = x_pred[:, 0:1]
    u_pred = x_pred[:, 1:2]

    length = a_pred.shape[-1]
    h = 1 / (length - 1)
    u_padded = F.pad(u_pred, (1, 1, 1, 1), mode="constant", value=0)
    laplacian = (
        u_padded[..., :-2, 1:-1]
        + u_padded[..., 2:, 1:-1]
        + u_padded[..., 1:-1, :-2]
        + u_padded[..., 1:-1, 2:]
        - 4 * u_pred
    ) / h**2
    return (laplacian - a_pred)[..., 1:-1, 1:-1]


def _get_pde_residual(dataset_name: str):
    if dataset_name == "darcy":
        return _get_darcy_residual
    if dataset_name == "poisson":
        return _get_poisson_residual
    raise ValueError(f"Unsupported canonical dataset: {dataset_name}")


class _Observation:
    def __init__(self, config: dict[str, object], dataset_name: str):
        self.dataset_name = dataset_name
        self.config = config
        self.type = config["type"]
        self.loss_type = config["loss_type"]
        self.loss_func = _get_loss_func(self.loss_type)

    def init(self, ground_truth: torch.Tensor):
        raise NotImplementedError

    def get_observation_loss(self, x_pred: torch.Tensor):
        raise NotImplementedError

    def _calculate_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        n_obs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = pred - gt if mask is None else (pred - gt) * mask
        return self.loss_func(residual, n_obs)


class _SparseObservation(_Observation):
    """Marker base for dataset-specific fixed-coordinate observations."""


class _PDEObservation(_Observation):
    def __init__(self, config: dict[str, object], dataset_name: str):
        super().__init__(config, dataset_name)
        assert (
            self.config["derivative_method"] == "finite_diff"
        ), "Only finite difference method is supported"
        self.n_channels = 1
        self.pde_residual_func = _get_pde_residual(dataset_name)

    def init(
        self,
        ground_truth: torch.Tensor,
        normalizer: ModelNormalizer | None = None,
    ) -> None:
        self.device = ground_truth.device
        self.resolution = ground_truth.shape[-1]
        self.ground_truth = ground_truth

    def get_observation_loss(self, x_pred: torch.Tensor) -> torch.Tensor:
        pde_residual = self.pde_residual_func(x_pred)
        n_obs = pde_residual.shape[-1] ** 2
        loss = self.loss_func(pde_residual, n_obs)
        return loss.sum(dim=1, keepdim=True)


class _PDESolver:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda")
        self.net = None
        self.noise_sampler = None
        self.batch_size = int(config["batch_size"])
        self.spatial_shape = as_spatial_shape(config["resolution"])
        self.resolution = self.spatial_shape[0]
        self.num_steps = int(config["iterations"])
        self.save_indices = None

    def generate_latents(self) -> torch.Tensor:
        if self.config["init_latents"] == "white_noise":
            return torch.randn(
                [self.batch_size, self.n_channels, *self.spatial_shape],
                device=self.device,
            )
        if self.config["init_latents"] == "rbf":
            if self.noise_sampler is None:
                raise RuntimeError("RBF latent generation requires a noise sampler")
            return self.noise_sampler.sample(self.batch_size)
        raise ValueError(f"Invalid init_latents value: {self.config['init_latents']}")

    def generate_single_batch(self, observations, class_labels=None):
        raise NotImplementedError


class _PDESolverDPS(_PDESolver):
    def __init__(self, config: Config):
        super().__init__(config)
        self.sigma_min = config["sigma_min"]
        self.sigma_max = config["sigma_max"]
        self.rho = config["rho"]
        self.weights = config["guidance"]["weights"]

    def generate_single_batch(self, observations, class_labels=None):
        latents = self.generate_latents()

        step_indices = torch.arange(
            self.num_steps,
            dtype=torch.float64,
            device=self.device,
        )
        sigma_t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        sigma_t_steps = torch.cat(
            [
                self.net.round_sigma(sigma_t_steps),
                torch.zeros_like(sigma_t_steps[:1]),
            ]
        )
        self.sigma_t_steps = sigma_t_steps

        x_next = latents.to(torch.float64) * sigma_t_steps[0]
        intermediates = []
        loss_history = []

        steps = zip(sigma_t_steps[:-1], sigma_t_steps[1:])
        for i, (sigma_t_cur, sigma_t_next) in enumerate(
            tqdm(steps, total=self.num_steps)
        ):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            sigma_t = self.net.round_sigma(sigma_t_cur)

            x_N = self.net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            if i < self.num_steps - 1:
                x_N = self.net(x_next, sigma_t_next, class_labels=class_labels).to(
                    torch.float64
                )
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

            denorm_x_N = self.normalizer.denormalize(x_N)
            denorm_x_next = self.normalizer.denormalize(x_next.detach())

            update = torch.zeros_like(x_cur)
            if i < self.num_steps - 1:
                weight_ptr = 0
                step_losses = []
                active_losses = []

                for obs in observations:
                    loss = obs.get_observation_loss(denorm_x_N)
                    step_losses.append(loss.detach())

                    coef = self.get_coef(cur_step=i, obs_type=obs.type)
                    for c in range(obs.n_channels):
                        if coef != 0 and self.weights[weight_ptr] != 0:
                            active_losses.append(
                                (loss[:, c].sum(), coef * self.weights[weight_ptr])
                            )
                        weight_ptr += 1

                for idx, (loss, weight) in enumerate(active_losses):
                    retain_graph = idx < len(active_losses) - 1
                    grad = torch.autograd.grad(
                        loss,
                        x_cur,
                        retain_graph=retain_graph,
                    )[0]
                    update = update + weight * grad

                loss_history.append(torch.cat(step_losses, dim=1))

            if getattr(self, "project_gradient", None) is not None:
                update = self.project_gradient(x_next, update)

            x_next = x_next - update
            if x_next.isnan().any():
                print(f"\nStep {i}: NaN detected!")
                break

            if self.save_indices is not None and i in self.save_indices:
                denorm_x_updated = self.normalizer.denormalize(x_next.detach())
                intermediates.append(
                    torch.cat(
                        [denorm_x_N.detach(), denorm_x_next, denorm_x_updated],
                        dim=1,
                    )
                )

        x_final = x_next.detach()
        pred = self.normalizer.transform(x_final, denormalize=True)
        aux = {"intermediates": intermediates, "loss_history": loss_history}
        return pred, aux

    def get_coef(self, cur_step: int, obs_type: str) -> float:
        if self.sigma_t_steps[cur_step] > 1.0:
            return 0 if obs_type == "pde" else 1
        return self.sigma_t_steps[cur_step].item()


class FieldSparseObservation(_SparseObservation):
    """Sparse conditioning for one observed model field only."""

    def __init__(
        self,
        config: dict[str, object],
        dataset_name: str,
        field: str,
        channel_index: int,
    ):
        super().__init__(config, dataset_name)
        self.field = field
        self.channel_index = channel_index
        self.n_channels = 1

    def init_from_coords(
        self,
        observed_field: torch.Tensor,
        sensor_coords: np.ndarray,
        normalizer: ModelNormalizer | None = None,
    ) -> None:
        if observed_field.ndim != 4 or observed_field.shape[1] != 1:
            raise ValueError(
                "FieldSparseObservation expects observed values with shape [N,1,H,W]"
            )
        self.device = observed_field.device
        self.spatial_shape = tuple(int(value) for value in observed_field.shape[-2:])

        coords = torch.as_tensor(sensor_coords, device=self.device, dtype=torch.long)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
            raise ValueError("sensor_coords must have non-empty shape [N,2]")
        height, width = self.spatial_shape
        if (
            torch.any(coords[:, 0] < 0)
            or torch.any(coords[:, 0] >= height)
            or torch.any(coords[:, 1] < 0)
            or torch.any(coords[:, 1] >= width)
        ):
            raise ValueError("sensor_coords are outside the observation field")

        mask = torch.zeros(
            (1, height, width),
            device=self.device,
            dtype=observed_field.dtype,
        )
        mask[0, coords[:, 0], coords[:, 1]] = 1
        self.masks = mask
        n_obs = int(mask.sum().item())
        self.n_obs_list = [n_obs]
        self.known_indices = torch.tensor(
            [[n_obs]], device=self.device, dtype=observed_field.dtype
        )

        self.to_normalize = bool(self.config["normalize"])
        self.normalizer = normalizer
        self.ground_truth = observed_field
        if self.to_normalize:
            if normalizer is None:
                raise ValueError("A normalizer is required for normalized observations")
            self.ground_truth = normalizer.normalize_channel(observed_field, self.field)
        self.interpolation_mode = None

    def get_observation_loss(self, x_pred: torch.Tensor) -> torch.Tensor:
        prediction = x_pred[:, self.channel_index : self.channel_index + 1]
        if self.to_normalize:
            prediction = self.normalizer.normalize_channel(prediction, self.field)
        if self.interpolation_mode is not None and prediction.shape[-2:] != self.spatial_shape:
            prediction = F.interpolate(
                prediction,
                size=self.spatial_shape,
                mode=self.interpolation_mode,
                align_corners=True,
            )
        return self._calculate_loss(
            prediction,
            self.ground_truth,
            self.known_indices,
            self.masks,
        )


class FieldAreaAverageObservation(_SparseObservation):
    """Condition one model field on a complete lower-resolution observation."""

    def __init__(
        self,
        config: dict[str, object],
        dataset_name: str,
        field: str,
        channel_index: int,
    ):
        super().__init__(config, dataset_name)
        self.field = field
        self.channel_index = channel_index
        self.n_channels = 1

    def init_from_observation(
        self,
        observed_field: torch.Tensor,
        model_resolution: object,
        normalizer: ModelNormalizer | None = None,
    ) -> None:
        if observed_field.ndim != 4 or observed_field.shape[1] != 1:
            raise ValueError(
                "FieldAreaAverageObservation expects values with shape [N,1,H,W]"
            )
        if observed_field.shape[0] != 1:
            raise ValueError("Benchmark observations must contain exactly one field")
        observation_height, observation_width = observed_field.shape[-2:]
        if observation_height <= 0 or observation_width <= 0:
            raise ValueError("Observation resolution must be positive")
        model_shape = as_spatial_shape(model_resolution)
        if (
            model_shape[0] % observation_height != 0
            or model_shape[1] % observation_width != 0
        ):
            raise ValueError(
                f"Model resolution {model_shape} must be divisible by "
                f"observation shape {(observation_height, observation_width)}"
            )

        self.device = observed_field.device
        self.observation_size = (observation_height, observation_width)
        self.known_indices = torch.tensor(
            [[observation_height * observation_width]],
            device=self.device,
            dtype=observed_field.dtype,
        )
        self.to_normalize = bool(self.config["normalize"])
        self.normalizer = normalizer
        self.ground_truth = observed_field
        if self.to_normalize:
            if normalizer is None:
                raise ValueError("A normalizer is required for normalized observations")
            self.ground_truth = normalizer.normalize_channel(observed_field, self.field)

    def get_observation_loss(self, x_pred: torch.Tensor) -> torch.Tensor:
        prediction = x_pred[:, self.channel_index : self.channel_index + 1]
        prediction = F.adaptive_avg_pool2d(prediction, self.observation_size)
        if self.to_normalize:
            prediction = self.normalizer.normalize_channel(prediction, self.field)
        return self._calculate_loss(
            prediction,
            self.ground_truth,
            self.known_indices,
        )


def _resize_fields(
    fields: dict[str, np.ndarray],
    channels: tuple[str, ...],
    resolution: object,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(
        np.stack([fields[channel] for channel in channels], axis=0)
    ).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    spatial_shape = as_spatial_shape(resolution)
    if tensor.shape[-2:] != spatial_shape:
        tensor = F.interpolate(
            tensor,
            size=spatial_shape,
            mode="bilinear",
            align_corners=False,
        )
    return tensor


class FunDPSAdapter(MethodAdapter):
    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        if self.profile.method != "fundps":
            raise ValueError(f"FunDPS adapter cannot use method profile '{self.profile.method}'")
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match '{self.dataset.name}'"
            )
        self.dataset.validate_model_channels(self.profile.channels)
        if bool(self.profile.guidance.get("pending", False)):
            raise ValueError(
                f"Guidance is not finalized for dataset '{self.dataset.name}'; "
                "run only --dry-run until a real checkpoint smoke test fixes it"
            )

        self.torch_device = torch.device(self.device)
        install_legacy_checkpoint_aliases()
        checkpoint = load_pickle_with_torch_storage_map(
            self.checkpoint,
            map_location=torch.device("cpu"),
        )
        if "ema" not in checkpoint:
            raise KeyError(f"FunDPS checkpoint {self.checkpoint} has no 'ema' network")
        self.net = checkpoint["ema"].to(self.torch_device)
        self.net.eval()

        checkpoint_shape = as_spatial_shape(self.net.img_resolution)
        checkpoint_resolution = spatial_to_config(checkpoint_shape)
        requested_resolution = self.profile.resolution
        if int(self.net.img_channels) != len(self.profile.channels):
            raise ValueError(
                f"Checkpoint has {self.net.img_channels} channels, profile has "
                f"{len(self.profile.channels)}"
            )
        self.checkpoint_resolution = checkpoint_resolution
        self.model_shape = as_spatial_shape(requested_resolution or checkpoint_resolution)
        self.model_resolution = spatial_to_config(self.model_shape)

        sampling = self.profile.sampling
        guidance = self.profile.guidance
        config = Config(
            {
                "outdir": ".",
                "batch_size": 1,
                "resolution": self.model_resolution,
                "iterations": int(sampling.get("iterations", 500)),
                "n_plots": 0,
                "n_process_steps": 1,
                "observation": [],
                "sigma_min": float(sampling.get("sigma_min", 0.002)),
                "sigma_max": float(sampling.get("sigma_max", 80.0)),
                "rho": float(sampling.get("rho", 7.0)),
                "guidance": {"weights": []},
            }
        )
        self.solver = _PDESolverDPS(config)
        self.solver.device = self.torch_device
        self.solver.net = self.net
        self.solver.n_channels = int(self.net.img_channels)
        self.solver.save_indices = None

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        self.solver.normalizer = self.normalizer
        self.rbf_scale = float(sampling.get("rbf_scale", 0.05))
        self.init_latents = str(sampling.get("init_latents", "rbf"))
        self.solver.config.update("init_latents", self.init_latents)
        self.solver.config.update("rbf_scale", self.rbf_scale)

        field_weights = guidance.get("field_weights", {})
        unexpected_weights = {
            channel: float(weight)
            for channel, weight in field_weights.items()
            if channel != self.dataset.observed_field and float(weight) != 0
        }
        if unexpected_weights:
            raise ValueError(
                "FunDPS benchmark conditioning may only weight the observed field "
                f"'{self.dataset.observed_field}', got {unexpected_weights}"
            )
        self.observation_weight = float(
            field_weights.get(self.dataset.observed_field, 0.0)
        )
        self.pde_weight = float(guidance.get("pde_weight", 0.0))
        self.observation_loss = str(guidance.get("loss_type", "mse"))
        self.normalize_observation = bool(guidance.get("normalize", True))

    def _build_observations(
        self,
        case: BenchmarkCase,
    ) -> list[object]:
        observed_channel = self.profile.channels.index(case.observed_field)
        observation_config = {
            "type": "sparse",
            "loss_type": self.observation_loss,
            "normalize": self.normalize_observation,
        }
        observed = torch.from_numpy(
            case.fields[case.observed_field]
        ).unsqueeze(0).unsqueeze(0).to(
            device=self.torch_device,
            dtype=torch.float32,
        )

        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = scale_sensor_coords(
                case.sensor_coords,
                case.observation_resolution,
                self.model_shape,
                unique=True,
            )
            if observed.shape[-2:] != self.model_shape:
                observed = F.interpolate(
                    observed,
                    size=self.model_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            sparse = FieldSparseObservation(
                observation_config,
                self.dataset.name,
                field=case.observed_field,
                channel_index=observed_channel,
            )
            sparse.init_from_coords(
                observed,
                coords,
                self.normalizer,
            )
            observation: object = sparse
        elif case.observation_operator == "area_average":
            area_average = FieldAreaAverageObservation(
                observation_config,
                self.dataset.name,
                field=case.observed_field,
                channel_index=observed_channel,
            )
            area_average.init_from_observation(
                observed,
                self.model_resolution,
                self.normalizer,
            )
            observation = area_average
        else:
            raise ValueError(
                f"Unsupported observation operator: {case.observation_operator}"
            )

        observations: list[object] = [observation]
        weights = [self.observation_weight]

        if self.pde_weight != 0:
            ground_truth = _resize_fields(
                case.fields,
                self.profile.channels,
                self.model_shape,
                self.torch_device,
            )
            pde_config = {
                "type": "pde",
                "loss_type": str(self.profile.guidance.get("pde_loss_type", "huber")),
                "derivative_method": "finite_diff",
            }
            pde = _PDEObservation(pde_config, self.dataset.name)
            pde.init(ground_truth)
            observations.append(pde)
            weights.append(self.pde_weight)

        self.solver.weights = weights
        return observations

    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        observations = self._build_observations(case)

        if self.init_latents == "rbf":
            self.solver.noise_sampler = RBFKernel(
                len(self.profile.channels),
                self.model_shape[0],
                self.model_shape[1],
                scale=self.rbf_scale,
                device=self.torch_device,
            )
        else:
            self.solver.noise_sampler = None

        generated: list[torch.Tensor] = []
        count = 0
        while count < num_samples:
            current_batch = min(batch_size, num_samples - count)
            self.solver.batch_size = current_batch
            prediction, _ = self.solver.generate_single_batch(observations)
            prediction = prediction.to(torch.float32)
            if prediction.shape[-2:] != case.resolution:
                prediction = F.interpolate(
                    prediction,
                    size=case.resolution,
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = self.normalizer.apply_postprocess(prediction)
            generated.append(prediction.cpu())
            count += current_batch

        samples = torch.cat(generated, dim=0)[:num_samples].numpy()
        return {
            channel: samples[:, index]
            for index, channel in enumerate(self.profile.channels)
        }
