from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _device_get_tree(tree: Any) -> Any:
    try:
        import jax
    except ImportError:
        return tree
    return jax.device_get(tree)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_payload = _device_get_tree(payload)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(cpu_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FunDiff checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"FunDiff checkpoint must be a dictionary: {checkpoint_path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"FunDiff metadata not found: {json_path}")
    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"FunDiff metadata must be a JSON object: {json_path}")
    return payload


def package_paths(package_dir: str | Path) -> dict[str, Path]:
    root = Path(package_dir)
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "target_fae": root / "target_fae.pkl",
        "condition_fae": root / "condition_fae.pkl",
        "dit": root / "dit.pkl",
    }

