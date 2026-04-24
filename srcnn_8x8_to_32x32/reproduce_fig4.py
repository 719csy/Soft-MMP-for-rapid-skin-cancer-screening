from __future__ import annotations

import argparse
from pathlib import Path

from .infer import run_inference


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the public toy figure-pipeline placeholder for the SRCNN path."
    )
    parser.add_argument("--config", type=Path, default=package_dir / "config.paper.yaml")
    parser.add_argument("--out_dir", type=Path, default=Path("release/toy_outputs"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    outputs = run_inference(
        config_path=args.config,
        input_path=package_dir / "examples" / "toy_input_8x8.csv",
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint,
        output_prefix="toy_fig4_placeholder_32x32",
        device_name=args.device,
        require_checkpoint=False,
    )
    print("This is a public toy placeholder, not a final manuscript figure reproduction.")
    print(f"Wrote {outputs['png']}")


if __name__ == "__main__":
    main()

