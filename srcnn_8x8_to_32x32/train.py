from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import dataset_summary, load_manifest, split_records
from .infer import build_model, load_checkpoint, min_max_normalize
from .io_utils import load_config, read_grid_csv, write_json
from .metrics import metric_dict, negative_psnr_loss
from .provenance import environment, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRCNN with grouped 8x8 inputs and 32x32 targets.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    for name, kind in (("epochs", int), ("batch_size", int), ("lr", float), ("weight_decay", float), ("seed", int)):
        parser.add_argument("--" + name, type=kind, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def compute_norm_stats(records: list[dict], input_size: int, target_size: int, target_field: str) -> dict[str, float]:
    bounds = {"input_min": float("inf"), "input_max": float("-inf"),
              "target_min": float("inf"), "target_max": float("-inf")}
    for record in records:
        for name, size, field in (("input", input_size, None), ("target", target_size, target_field)):
            grid = read_grid_csv(record[name + "_path"], (size, size), value_column=field)
            bounds[name + "_min"] = min(bounds[name + "_min"], float(grid.min()))
            bounds[name + "_max"] = max(bounds[name + "_max"], float(grid.max()))
    return bounds


class SrcnnCsvDataset(Dataset):
    def __init__(self, records: list[dict], config: dict, norm_stats: dict):
        self.records, self.config, self.norm_stats = records, config, norm_stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        cfg = self.config["data"]
        result = {"sample_id": record["sample_id"], "group_id": record["group_id"]}
        for name, size, field in (("input", 8, None), ("target", 32, cfg.get("target_field", "mean_kpa"))):
            grid = read_grid_csv(record[name + "_path"], (size, size), value_column=field)
            normalized = min_max_normalize(grid, self.norm_stats[name + "_min"], self.norm_stats[name + "_max"])
            result[name] = torch.from_numpy(normalized).unsqueeze(0).float()
        return result


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch["input"].to(device))
            targets = batch["target"].to(device)
            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append({"sample_id": sample_id, "group_id": batch["group_id"][i],
                             **metric_dict(predictions[i:i+1], targets[i:i+1])})
    if not rows:
        return {}
    names = ("psnr", "ssim", "mae", "rmse")
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    group_metrics = {group: {key: float(np.mean([r[key] for r in members])) for key in names}
                     for group, members in groups.items()}
    return {
        "sample_mean": {key: float(np.mean([r[key] for r in rows])) for key in names},
        "group_mean": {key: float(np.mean([r[key] for r in group_metrics.values()])) for key in names},
        "groups": group_metrics, "samples": rows,
        "scale": "Training-normalized targets; PSNR data range 1; no clipping",
    }


def run_training(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    training = config.get("training", {})
    defaults = {"seed": 42, "epochs": 200, "batch_size": 4, "lr": 1e-4, "weight_decay": 0.0}
    settings = {key: getattr(args, key, None) if getattr(args, key, None) is not None
                else training.get(key, value) for key, value in defaults.items()}
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Epochs, batch size and learning rate must be positive; weight decay must be nonnegative")
    config["training"] = {**training, **settings, "loss": "negative_psnr"}
    smoke_test = bool(getattr(args, "smoke_test", False))
    set_seed(settings["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    records = load_manifest(args.manifest, config)
    train_records, val_records, test_records = split_records(records, smoke_test)
    stats = compute_norm_stats(train_records, 8, 32, config["data"].get("target_field", "mean_kpa"))
    train_loader = DataLoader(SrcnnCsvDataset(train_records, config, stats),
                              batch_size=settings["batch_size"], shuffle=True)
    # Held-out maps are evaluated individually. Group means give each independent unit equal weight.
    val_loader = DataLoader(SrcnnCsvDataset(val_records, config, stats), batch_size=1, shuffle=False)
    summary = {"run_kind": "smoke_test" if smoke_test else "user_dataset_training",
               "manuscript_checkpoint": False, "dataset": dataset_summary(records),
               "manifest_sha256": sha256(args.manifest), "config_sha256": sha256(args.config),
               "effective_config": config, "norm_stats": stats, "environment": environment(),
               "device": str(device), "selection": "validation group-mean PSNR" if val_records else "none; smoke test last epoch",
               "normalization": "Training min/max; no clipping"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse reuse so old metrics/checkpoints cannot be mistaken for this run.
    if any(args.out_dir.iterdir()):
        raise FileExistsError("Output directory must be empty; choose a new run directory")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best_score, best_epoch = float("-inf"), None
    best_path, last_path = args.out_dir / "best.pt", args.out_dir / "last.pt"
    with (args.out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = negative_psnr_loss(model(batch["input"].to(device)), batch["target"].to(device))
                loss.backward()
                optimizer.step()
                losses.append((float(loss.item()), len(batch["sample_id"])))
            val = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "negative_psnr_loss": sum(x*n for x,n in losses)/sum(n for _,n in losses), "val": val}
            log.write(json.dumps(row) + "\n")
            log.flush()
            checkpoint = {"model_state": model.state_dict(), "norm_stats": stats, "config": config,
                          "epoch": epoch, "provenance": summary}
            torch.save(checkpoint, last_path)
            if val and val["group_mean"]["psnr"] > best_score:
                best_score, best_epoch = val["group_mean"]["psnr"], epoch
                torch.save(checkpoint, best_path)
    chosen = best_path if val_records else last_path
    model.load_state_dict(load_checkpoint(chosen, device)["model_state"])
    summary["selected_epoch"] = best_epoch if val_records else settings["epochs"]
    summary["selected_checkpoint"] = chosen.name
    summary["checkpoint_sha256"] = sha256(chosen)
    summary["test_evaluated"] = bool(test_records) and not smoke_test
    # Test split is used only once, after loading the best validation checkpoint.
    if summary["test_evaluated"]:
        test_loader = DataLoader(SrcnnCsvDataset(test_records, config, stats), batch_size=1, shuffle=False)
        write_json(args.out_dir / "test_metrics.json", evaluate(model, test_loader, device))
    write_json(args.out_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    result = run_training(parse_args())
    print(f"Completed {result['run_kind']}: {result['dataset']['sample_count']} maps, "
          f"{result['dataset']['independent_group_count']} groups; manuscript checkpoint: false.")


if __name__ == "__main__":
    main()
