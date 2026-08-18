from posteriorbench.datasets.base import DatasetAdapter


class LightTransportDataset(DatasetAdapter):
    name = "light_transport"
    target_fields = ("sigma_t1", "sigma_t2")
    observed_field = "u"
    observation_operator = "area_average"

    def validate_model_channels(self, channels: tuple[str, ...]) -> None:
        expected = ("sigma_t1", "sigma_t2", "u")
        if channels != expected:
            raise ValueError(
                f"Light Transport expects model channels {expected}, got {channels}"
            )
