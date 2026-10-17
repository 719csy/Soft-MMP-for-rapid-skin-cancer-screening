from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .io_utils import load_config, read_grid_csv, save_heatmap, write_grid_csv, write_json
from .model import SRCNN8x8To32x32
from .provenance import environment, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRCNN inference from 8x8 Hall CSV to 32x32 map.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="8x8 Hall input CSV.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint including training normalization and provenance.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_checkpoint", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Explicitly permit random weights for an interface check.")
    return parser.parse_args()


def min_max_normalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scale = max(float(maximum) - float(minimum), 1e-8)
    return (array.astype(np.float32) - float(minimum)) / scale


def min_max_denormalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return array.astype(np.float32) * (float(maximum) - float(minimum)) + float(minimum)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model_state", "norm_stats", "config", "provenance"}.issubset(checkpoint):
        raise ValueError("Checkpoint must include model_state, norm_stats, config and provenance")
    stats = checkpoint["norm_stats"]
    for prefix in ("input", "target"):
        low, high = stats.get(prefix + "_min"), stats.get(prefix + "_max")
        if low is None or high is None or not np.isfinite([low, high]).all() or high < low:
            raise ValueError("Checkpoint contains invalid training normalization statistics")
    return checkpoint


def build_model(config: dict) -> SRCNN8x8To32x32:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    return SRCNN8x8To32x32(
        input_size=int(data_cfg.get("input_size", 8)),
        target_size=int(data_cfg.get("target_size", 32)),
        channels=tuple(model_cfg.get("channels", [64, 32])),
    )


def run_inference(
    config_path: Path,
    input_path: Path,
    out_dir: Path,
    checkpoint_path: Path | None = None,
    output_prefix: str | None = None,
    device_name: str | None = None,
    require_checkpoint: bool = False,
    seed: int = 42,
    smoke_test: bool = False,
) -> dict:
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    input_size = int(data_cfg.get("input_size", 8))
    target_size = int(data_cfg.get("target_size", 32))
    output_prefix = output_prefix or inference_cfg.get("output_prefix", "prediction_32x32")
    if checkpoint_path is None and (require_checkpoint or not smoke_test):
        raise FileNotFoundError("A checkpoint is required. Use --smoke_test only for a random-weight interface check.")

    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    norm_stats = None
    checkpoint_provenance = None

    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        saved_config = checkpoint["config"]
        if saved_config.get("data") != config.get("data") or saved_config.get("model") != config.get("model"):
            raise ValueError("Inference data/model configuration must match the checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        norm_stats = checkpoint["norm_stats"]
        checkpoint_provenance = checkpoint["provenance"]

    grid = read_grid_csv(input_path, expected_shape=(input_size, input_size))
    if norm_stats:
        input_norm = min_max_normalize(grid, norm_stats["input_min"], norm_stats["input_max"])
    else:
        input_norm = min_max_normalize(grid, float(grid.min()), float(grid.max()))

    tensor = torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        prediction_norm = model(tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    if norm_stats and {"target_min", "target_max"}.issubset(norm_stats):
        prediction = min_max_denormalize(prediction_norm, norm_stats["target_min"], norm_stats["target_max"])
    else:
        prediction = prediction_norm

    if prediction.shape != (target_size, target_size):
        raise RuntimeError(f"Expected {(target_size, target_size)} output, got {prediction.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{output_prefix}.csv"
    png_path = out_dir / f"{output_prefix}.png"
    metadata_path = out_dir / "inference_metadata.json"
    write_grid_csv(csv_path, prediction)
    is_demo = checkpoint_path is None or checkpoint_provenance.get("run_kind") == "smoke_test"
    label = "DEMO ONLY" if is_demo else "User checkpoint; figure provenance unverified"
    units = data_cfg.get("target_units", "target units") if norm_stats else "Normalized output (a.u.)"
    save_heatmap(png_path, prediction, title=f"{label}\n32x32 reconstruction from 8x8 array", units=units)
    input_png = out_dir / f"{output_prefix}_input_8x8.png"
    save_heatmap(input_png, grid, title=f"{label}\n8x8 sensor-array input", units=data_cfg.get("input_units", "input units"))
    write_json(
        metadata_path,
        {
            "config": str(config_path),
            "input": str(input_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "output_csv": str(csv_path),
            "output_png": str(png_path),
            "input_size": input_size,
            "target_size": target_size,
            "checkpoint_loaded": checkpoint_path is not None,
            "run_kind": "smoke_test" if is_demo else "checkpoint_inference",
            "manuscript_figure_reproduction": False,
            "checkpoint_provenance": checkpoint_provenance,
            "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None,
            "input_sha256": sha256(input_path),
            "config_sha256": sha256(config_path),
            "norm_stats": norm_stats,
            "normalization": "Training min/max; no clipping" if norm_stats else "Per-input min/max; smoke test only",
            "output_units": units,
            "input_heatmap": str(input_png),
            "seed": seed,
            "device": str(device),
            "environment": environment(),
        },
    )
    return {"csv": csv_path, "png": png_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    outputs = run_inference(
        config_path=args.config,
        input_path=args.input,
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint,
        output_prefix=args.output_prefix,
        device_name=args.device,
        require_checkpoint=args.require_checkpoint,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    print(f"Wrote {outputs['csv']}")
    print(f"Wrote {outputs['png']}")


if __name__ == "__main__":
    main()

