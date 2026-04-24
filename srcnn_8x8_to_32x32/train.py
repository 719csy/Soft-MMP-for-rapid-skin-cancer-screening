from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .infer import build_model, min_max_normalize
from .io_utils import load_config, read_grid_csv, resolve_relative, save_heatmap, write_grid_csv, write_json
from .metrics import metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRCNN on 8x8 Hall CSV inputs and 32x32 targets.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.paper.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_norm_stats(records: list[dict], input_size: int, target_size: int, target_field: str) -> dict[str, float]:
    input_min = float("inf")
    input_max = float("-inf")
    target_min = float("inf")
    target_max = float("-inf")
    for record in records:
        input_grid = read_grid_csv(record["input_path"], expected_shape=(input_size, input_size))
        target_grid = read_grid_csv(
            record["target_path"],
            expected_shape=(target_size, target_size),
            value_column=target_field,
        )
        input_min = min(input_min, float(input_grid.min()))
        input_max = max(input_max, float(input_grid.max()))
        target_min = min(target_min, float(target_grid.min()))
        target_max = max(target_max, float(target_grid.max()))
    return {
        "input_min": input_min,
        "input_max": input_max,
        "target_min": target_min,
        "target_max": target_max,
    }


def load_manifest(path: Path, config: dict) -> list[dict]:
    data_cfg = config.get("data", {})
    input_column = data_cfg.get("input_column", "input_8x8_csv")
    target_column = data_cfg.get("target_column", "target_32x32_csv")
    frame = pd.read_csv(path)
    missing = [column for column in (input_column, target_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    base_dir = path.parent
    records = []
    for index, row in frame.iterrows():
        records.append(
            {
                "sample_id": str(row.get("sample_id", f"sample_{index:04d}")),
                "input_path": resolve_relative(row[input_column], base_dir),
                "target_path": resolve_relative(row[target_column], base_dir),
                "split": str(row.get("split", "train")),
            }
        )
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    return records


class SrcnnCsvDataset(Dataset):
    def __init__(self, records: list[dict], config: dict, norm_stats: dict[str, float]):
        self.records = records
        self.config = config
        self.norm_stats = norm_stats
        data_cfg = config.get("data", {})
        self.input_size = int(data_cfg.get("input_size", 8))
        self.target_size = int(data_cfg.get("target_size", 32))
        self.target_field = data_cfg.get("target_field", "mean_kpa")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        input_grid = read_grid_csv(record["input_path"], expected_shape=(self.input_size, self.input_size))
        target_grid = read_grid_csv(
            record["target_path"],
            expected_shape=(self.target_size, self.target_size),
            value_column=self.target_field,
        )
        input_norm = min_max_normalize(input_grid, self.norm_stats["input_min"], self.norm_stats["input_max"])
        target_norm = min_max_normalize(target_grid, self.norm_stats["target_min"], self.norm_stats["target_max"])
        return {
            "sample_id": record["sample_id"],
            "input": torch.from_numpy(input_norm).unsqueeze(0).float(),
            "target": torch.from_numpy(target_norm).unsqueeze(0).float(),
        }


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    train_records = [record for record in records if record["split"].lower() == "train"]
    val_records = [record for record in records if record["split"].lower() in {"val", "valid", "validation"}]
    if not train_records:
        train_records = records
    return train_records, val_records


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            rows.append(metric_dict(model(inputs), targets))
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_training(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 200))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 0.0))
    data_cfg = config.get("data", {})
    input_size = int(data_cfg.get("input_size", 8))
    target_size = int(data_cfg.get("target_size", 32))
    target_field = data_cfg.get("target_field", "mean_kpa")

    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = load_manifest(args.manifest, config)
    train_records, val_records = split_records(records)
    norm_stats = compute_norm_stats(train_records, input_size, target_size, target_field)

    train_loader = DataLoader(
        SrcnnCsvDataset(train_records, config, norm_stats),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(SrcnnCsvDataset(val_records, config, norm_stats), batch_size=batch_size)
        if val_records
        else None
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"
    best_metric = float("-inf")
    best_path = args.out_dir / "best.pt"
    last_path = args.out_dir / "last.pt"

    with open(log_path, "w", encoding="utf-8") as log_handle:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = F.mse_loss(prediction, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device) if val_loader else {}
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else None,
                "train": train_metrics,
                "val": val_metrics,
            }
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()

            score = val_metrics.get("psnr", train_metrics.get("psnr", float("-inf")))
            checkpoint = {
                "model_state": model.state_dict(),
                "norm_stats": norm_stats,
                "config": config,
                "epoch": epoch,
            }
            torch.save(checkpoint, last_path)
            if score >= best_metric:
                best_metric = score
                torch.save(checkpoint, best_path)

    first_record = records[0]
    preview_grid = read_grid_csv(first_record["input_path"], expected_shape=(input_size, input_size))
    preview_input = min_max_normalize(preview_grid, norm_stats["input_min"], norm_stats["input_max"])
    model.eval()
    with torch.no_grad():
        preview = model(torch.from_numpy(preview_input).unsqueeze(0).unsqueeze(0).float().to(device))
    preview = preview.squeeze(0).squeeze(0).cpu().numpy()
    write_grid_csv(args.out_dir / "preview_prediction_32x32.csv", preview)
    save_heatmap(args.out_dir / "preview_prediction_32x32.png", preview, "SRCNN smoke-test preview")

    summary = {
        "manifest": str(args.manifest),
        "sample_count": len(records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "epochs": epochs,
        "device": str(device),
        "norm_stats": norm_stats,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    write_json(args.out_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    summary = run_training(parse_args())
    print(f"Completed SRCNN smoke training on {summary['sample_count']} sample(s).")


if __name__ == "__main__":
    main()

