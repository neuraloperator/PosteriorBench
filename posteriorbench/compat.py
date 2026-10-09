import importlib
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def install_legacy_checkpoint_aliases() -> None:
    """Expose legacy FunDPS module names used by pickled checkpoints."""
    aliases = {
        "torch_utils": "models.torch_utils",
        "torch_utils.persistence": "models.torch_utils.persistence",
        "torch_utils.misc": "models.torch_utils.misc",
        "dnnlib": "models.dnnlib",
        "dnnlib.util": "models.dnnlib.util",
        "training": "training_fundps",
        "training.augment": "training_fundps.augment",
        "training.dataset_utils": "training_fundps.dataset_utils",
        "training.loss": "training_fundps.loss",
        "training.networks": "training_fundps.networks",
        "training.noise_samplers": "training_fundps.noise_samplers",
        "training.training_loop": "training_fundps.training_loop",
    }
    for legacy_name, current_name in aliases.items():
        sys.modules.setdefault(legacy_name, importlib.import_module(current_name))


@contextmanager
def _torch_pickle_map_location(map_location: torch.device | str):
    original_restore = torch.serialization.default_restore_location
    target_location = str(torch.device(map_location))

    def restore_to_target(storage, _location):
        return original_restore(storage, target_location)

    torch.serialization.default_restore_location = restore_to_target
    try:
        yield
    finally:
        torch.serialization.default_restore_location = original_restore


def load_pickle_with_torch_storage_map(
    path: str | Path,
    map_location: torch.device | str = torch.device("cpu"),
) -> Any:
    """Load a pickle that embeds torch storages onto a requested device."""
    with _torch_pickle_map_location(map_location):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
