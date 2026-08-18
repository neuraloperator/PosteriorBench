from posteriorbench.datasets.base import DatasetAdapter


class DarcyDataset(DatasetAdapter):
    name = "darcy"
    target_fields = ("a",)
    observed_field = "u"

    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        if channels != ("a", "u"):
            raise ValueError(f"Darcy expects model channels ('a', 'u'), got {channels}")
