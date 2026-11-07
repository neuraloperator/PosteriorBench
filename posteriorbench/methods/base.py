from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from posteriorbench.artifacts import CheckpointProfile
from posteriorbench.datasets.base import BenchmarkCase, DatasetAdapter
from posteriorbench.postprocess import postprocess_predictions


class MethodAdapter(ABC):
    def __init__(
        self,
        checkpoint: str | Path,
        profile: CheckpointProfile,
        dataset: DatasetAdapter,
        device: str,
    ):
        checkpoint_text = str(checkpoint)
        if checkpoint_text.startswith("hf://") or checkpoint_text.startswith(
            "https://huggingface.co/datasets/"
        ):
            self.checkpoint = checkpoint_text
        else:
            self.checkpoint = Path(checkpoint)
        self.profile = profile
        self.dataset = dataset
        self.device = device

    def finalize_samples(self, samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return postprocess_predictions(self.dataset.name, samples)

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError
