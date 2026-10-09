from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from posteriorbench.artifacts import CheckpointProfile
from posteriorbench.datasets.base import BenchmarkCase, DatasetAdapter


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
