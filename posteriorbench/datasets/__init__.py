from posteriorbench.datasets.base import BenchmarkCase, DatasetAdapter
from posteriorbench.datasets.ccs import CCSDataset
from posteriorbench.datasets.darcy import DarcyDataset
from posteriorbench.datasets.light_transport import LightTransportDataset
from posteriorbench.datasets.poisson import PoissonDataset


DATASETS: dict[str, type[DatasetAdapter]] = {
    "ccs": CCSDataset,
    "darcy": DarcyDataset,
    "light_transport": LightTransportDataset,
    "poisson": PoissonDataset,
}


def get_dataset(name: str) -> DatasetAdapter:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[key]()


__all__ = ["BenchmarkCase", "DatasetAdapter", "get_dataset"]
