from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_relative(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(base_dir) / path


def _numeric_array(frame: pd.DataFrame) -> np.ndarray | None:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.isna().any().any():
        return None
    return numeric.to_numpy(dtype=np.float32)


def _validate_shape(array: np.ndarray, expected_shape: tuple[int, int] | None, path: Path) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D CSV grid, got shape {array.shape}")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{path}: expected shape {expected_shape}, got {array.shape}")
    return array.astype(np.float32)


def read_grid_csv(
    path: str | Path,
    expected_shape: tuple[int, int] | None = None,
    value_column: str | None = None,
) -> np.ndarray:
    """Read matrix CSVs or long-table map CSVs into a float32 grid."""
    path = Path(path)

    frame = pd.read_csv(path)
    if value_column and {"x_norm", "y_norm", value_column}.issubset(frame.columns):
        grid = (
            frame.pivot(index="y_norm", columns="x_norm", values=value_column)
            .sort_index(axis=0)
            .sort_index(axis=1)
            .to_numpy(dtype=np.float32)
        )
        return _validate_shape(grid, expected_shape, path)

    attempts = [
        pd.read_csv(path, index_col=0),
        pd.read_csv(path, header=None),
        frame,
    ]
    for candidate in attempts:
        array = _numeric_array(candidate)
        if array is None:
            continue
        if expected_shape is None or tuple(array.shape) == tuple(expected_shape):
            return _validate_shape(array, expected_shape, path)

    expected = f" with shape {expected_shape}" if expected_shape else ""
    raise ValueError(f"{path}: could not read a numeric CSV grid{expected}")


def write_grid_csv(path: str | Path, array: np.ndarray, axis_min: float = -1.0, axis_max: float = 1.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    x_labels = np.linspace(axis_min, axis_max, array.shape[1], dtype=np.float32)
    y_labels = np.linspace(axis_min, axis_max, array.shape[0], dtype=np.float32)
    frame = pd.DataFrame(
        array,
        index=[f"{value:.6g}" for value in y_labels],
        columns=[f"{value:.6g}" for value in x_labels],
    )
    frame.index.name = "y_norm"
    frame.to_csv(path, float_format="%.8f")


def save_heatmap(path: str | Path, array: np.ndarray, title: str, cmap: str = "viridis") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.0, 3.6), dpi=180)
    image = axis.imshow(np.asarray(array), cmap=cmap)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.tick_params(labelsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

