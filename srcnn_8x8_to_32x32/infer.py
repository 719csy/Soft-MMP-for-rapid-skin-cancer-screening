from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .io_utils import load_config, read_grid_csv, save_heatmap, write_grid_csv, write_json
from .model import SRCNN8x8To32x32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRCNN inference from 8x8 Hall CSV to 32x32 map.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.paper.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="8x8 Hall input CSV.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional manuscript-matched checkpoint.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_checkpoint", action="store_true")
    return parser.parse_args()


def min_max_normalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scale = max(float(maximum) - float(minimum), 1e-8)
    return ((array.astype(np.float32) - float(minimum)) / scale).clip(0.0, 1.0)


def min_max_denormalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return array.astype(np.float32) * (float(maximum) - float(minimum)) + float(minimum)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
) -> dict:
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    input_size = int(data_cfg.get("input_size", 8))
    target_size = int(data_cfg.get("target_size", 32))
    output_prefix = output_prefix or inference_cfg.get("output_prefix", "prediction_32x32")
    allow_random_init = bool(inference_cfg.get("allow_random_init", False))

    if checkpoint_path is None and (require_checkpoint or not allow_random_init):
        raise FileNotFoundError("No checkpoint was provided. Set allow_random_init only for smoke tests.")

    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    norm_stats = None

    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        norm_stats = checkpoint.get("norm_stats")

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
    save_heatmap(png_path, prediction, title="SRCNN 32x32 prediction")
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
    )
    print(f"Wrote {outputs['csv']}")
    print(f"Wrote {outputs['png']}")


if __name__ == "__main__":
    main()

