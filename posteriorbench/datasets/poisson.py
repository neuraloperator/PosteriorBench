from posteriorbench.datasets.base import DatasetAdapter


class PoissonDataset(DatasetAdapter):
    name = "poisson"
    target_fields = ("f",)
    observed_field = "phi"

    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        if channels != ("f", "phi"):
            raise ValueError(f"Poisson expects model channels ('f', 'phi'), got {channels}")
