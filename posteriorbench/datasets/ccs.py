from posteriorbench.datasets.base import DatasetAdapter


class CCSDataset(DatasetAdapter):
    name = "ccs"
    target_fields = ("x",)
    observed_field = "y"
    observation_operator = "sparse_points"

    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        if channels != ("x", "y"):
            raise ValueError(f"CCS expects model channels ('x', 'y'), got {channels}")
