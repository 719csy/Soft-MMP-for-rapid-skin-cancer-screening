from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io_utils import save_heatmap, write_grid_csv, write_json


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    def bump(size):
        axis = np.linspace(-1, 1, size)
        x, y = np.meshgrid(axis, axis)
        return np.exp(-((x - 0.15)**2 + (y + 0.1)**2) / 0.18)
    # Analytic illustration only: no measurement, COMSOL export or physical calibration.
    signal = (0.2 + 0.8 * bump(8)).astype(np.float32)
    target = (80.0 + 120.0 * bump(32)).astype(np.float32)
    write_grid_csv(out_dir / "toy_input_8x8.csv", signal)
    write_grid_csv(out_dir / "toy_target_32x32.csv", target)
    (out_dir / "toy_manifest.csv").write_text(
        "sample_id,group_id,split,input_8x8_csv,target_32x32_csv\n"
        "toy_001,synthetic_pattern_001,train,toy_input_8x8.csv,toy_target_32x32.csv\n",
        encoding="utf-8")
    save_heatmap(out_dir / "toy_input_8x8.png", signal,
                 "SYNTHETIC EXAMPLE ONLY\n8x8 sensor-array input", units="Synthetic signal (a.u.)")
    write_json(out_dir / "toy_provenance.json", {
        "source": "generate_toy.py analytic Gaussian pattern", "measured_data": False,
        "sample_count": 1, "synthetic_pattern_count": 1,
        "biological_specimens": 0, "physical_phantoms": 0,
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "input_shape": [8, 8], "target_shape": [32, 32],
        "manuscript_dataset": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the one-pair synthetic interface example.")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).with_name("examples"))
    generate(parser.parse_args().out_dir)
