import torch


def transform_darcy(x: torch.Tensor) -> torch.Tensor:
    a, u = x[:, 0, :, :], x[:, 1, :, :]
    a = torch.where(a > 7.5, torch.tensor(12.0, device=x.device), torch.tensor(3.0, device=x.device))
    return torch.stack([a, u], dim=1)


class DatasetNormalizer:

    def __init__(self, dataset_name, stats):
        self.dataset_name = dataset_name
        mean = torch.as_tensor(stats["mean"])
        std = torch.as_tensor(stats["std"])
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("Normalizer mean/std must be matching one-dimensional arrays")
        if len(mean) == 0 or not torch.isfinite(mean).all():
            raise ValueError("Normalizer means must be non-empty and finite")
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("Normalizer standard deviations must be finite and positive")
        self.mean = mean.reshape(1, -1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1)
        self._transform = lambda x: x
        if dataset_name == "darcy":
            if len(mean) != 2:
                raise ValueError("Darcy normalization expects channels [a, u]")
            self._transform = transform_darcy

    def _check_shape(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W], got shape {tuple(x.shape)}")
        expected_channels = self.mean.shape[1]
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {x.shape[1]}"
            )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x_normalized = (x - self.mean) * (0.5 / self.std)
        return x_normalized

    def denormalize(self, x_normalized: torch.Tensor) -> torch.Tensor:
        self._check_shape(x_normalized)
        self.mean = self.mean.to(x_normalized.device)
        self.std = self.std.to(x_normalized.device)
        x = x_normalized / (0.5 / self.std) + self.mean
        return x

    def transform(self, x: torch.Tensor, denormalize=False) -> torch.Tensor:
        self._check_shape(x)
        if denormalize:
            x = self.denormalize(x)
        return self._transform(x)
