# SRCNN algorism

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EPS = 1e-8
DEFAULT_TARGET_ROOT = Path(r"H:\My Drive\Clincal_ontology\Image_export\lesions")
DEFAULT_BZ_ROOT = Path(r"H:\My Drive\Clincal_ontology\Image_export\lesions")
DEFAULT_OUT_DIR = Path(r"H:\My Drive\rigidity_interpolation\checkpoint_srcnn_5x5_delta_bz_64x64")
TARGET_FILE_MAP = {
    "mean_kpa": "interp_mean_kpa.csv",
    "mu_logE": "interp_mu_logE.csv",
}
STRAIN_PATTERNS = [
    re.compile(r"_s(?P<val>0(?:\.\d+)?)_bz$", re.IGNORECASE),
    re.compile(r"u[_\-]?load(?P<val>0(?:\.\d+)?)_bz$", re.IGNORECASE),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Paper-style SRCNN baseline for 5x5 delta-Bz to 64x64 modulus reconstruction."
    )
    parser.add_argument("--target_root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--bz_root", type=Path, default=DEFAULT_BZ_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target_field", choices=sorted(TARGET_FILE_MAP), default="mean_kpa")
    parser.add_argument("--baseline_strain", type=float, default=0.0)
    parser.add_argument("--delta_strain", type=float, default=0.08)
    parser.add_argument("--target_size", type=int, default=64)
    parser.add_argument("--hall_x_mm", type=str, default="1,3,5,7,9")
    parser.add_argument("--hall_y_mm", type=str, default="1,3,5,7,9")
    parser.add_argument("--sensor_z_mm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--disable_figures", action="store_true")
    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strain_label(value):
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_float_list(text):
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected a comma-separated float list, got {text!r}")
    return [float(item) for item in values]


def sample_group(sample_id):
    for prefix in ("midas_lesion_", "ph2_lesion_", "isic2018_lesion_"):
        if sample_id.startswith(prefix):
            return prefix.replace("_lesion_", "")
    return sample_id.split("_", 1)[0]


def make_sample_key(path, root):
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else path.parent.name


def extract_strain_label(path):
    stem = path.stem
    for rx in STRAIN_PATTERNS:
        match = rx.search(stem)
        if match:
            return strain_label(float(match.group("val")))
    raise ValueError(f"Cannot parse strain from filename: {path.name}")


def discover_target_samples(target_root, target_field):
    target_name = TARGET_FILE_MAP[target_field]
    samples = {}
    for sample_dir in sorted(target_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        target_path = sample_dir / target_name
        if target_path.exists():
            sample_id = sample_dir.name
            samples[sample_id] = {
                "sample_id": sample_id,
                "group": sample_group(sample_id),
                "target_path": target_path,
            }
    return samples


def discover_bz_samples(bz_root):
    samples = {}
    if not bz_root.exists():
        raise FileNotFoundError(f"BZ_ROOT does not exist: {bz_root}")
    for path in sorted(bz_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".csv"}:
            continue
        if "bz" not in path.name.lower():
            continue
        try:
            strain = extract_strain_label(path)
        except ValueError:
            continue
        sample_id = make_sample_key(path, bz_root)
        entry = samples.setdefault(
            sample_id,
            {"sample_id": sample_id, "group": sample_group(sample_id), "bz": {}},
        )
        entry["bz"][strain] = path
    return samples


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_pairing_artifacts(
    out_dir,
    target_root,
    bz_root,
    target_field,
    required_strains,
    paired,
    skipped,
):
    payload = {
        "target_root": str(target_root),
        "bz_root": str(bz_root),
        "target_field": target_field,
        "required_strains": list(required_strains),
        "paired_count": len(paired),
        "paired_ids": sorted(paired),
        "paired": {
            sample_id: {
                "group": rec["group"],
                "target": str(rec["target_path"]),
                "bz": {key: str(value) for key, value in rec["bz"].items()},
            }
            for sample_id, rec in sorted(paired.items())
        },
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    write_json(out_dir / "paired_samples.json", payload)


def pair_samples(target_samples, bz_samples, required_strains):
    paired = {}
    skipped = {}
    required_strains = tuple(required_strains)
    for sample_id, target_rec in sorted(target_samples.items()):
        bz_rec = bz_samples.get(sample_id)
        if bz_rec is None:
            skipped[sample_id] = {
                "group": target_rec["group"],
                "reason": "missing_bz_sample",
                "target": str(target_rec["target_path"]),
                "available_bz_strains": [],
            }
            continue
        available = sorted(bz_rec["bz"])
        missing = [strain for strain in required_strains if strain not in bz_rec["bz"]]
        if missing:
            skipped[sample_id] = {
                "group": target_rec["group"],
                "reason": f"missing_strains:{missing}",
                "target": str(target_rec["target_path"]),
                "available_bz_strains": available,
            }
            continue
        paired[sample_id] = {
            "sample_id": sample_id,
            "group": target_rec["group"],
            "target_path": target_rec["target_path"],
            "bz": {strain: bz_rec["bz"][strain] for strain in required_strains},
        }
    return paired, skipped


def read_numeric_table(path):
    df = pd.read_csv(path, comment="%", header=None, sep=r"[,\s]+", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"No numeric data found in {path}")
    return df.reset_index(drop=True)


def bilinear_sample_regular_grid(xs, ys, grid, query_xs, query_ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    qx = np.asarray(query_xs, dtype=np.float64)
    qy = np.asarray(query_ys, dtype=np.float64)

    ix = np.searchsorted(xs, qx, side="right") - 1
    iy = np.searchsorted(ys, qy, side="right") - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]

    wx = (qx - x0) / np.maximum(x1 - x0, 1e-12)
    wy = (qy - y0) / np.maximum(y1 - y0, 1e-12)

    v00 = grid[iy, ix]
    v10 = grid[iy, ix + 1]
    v01 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def resize_map(array, size, mode):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    out = F.interpolate(tensor, size=(size, size), **kwargs)
    return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def load_bz_to_5x5(path, hall_points_xy, sensor_z_mm):
    df = read_numeric_table(path)

    if df.shape == (5, 5):
        return df.to_numpy(dtype=np.float32)
    if df.shape[0] == 5 and df.shape[1] == 6:
        return df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if df.shape[1] < 3:
        raise ValueError(f"{path.name}: expected at least 3 numeric columns, got {df.shape}")

    if df.shape[1] == 3:
        x = df.iloc[:, 0].to_numpy(dtype=np.float64)
        y = df.iloc[:, 1].to_numpy(dtype=np.float64)
        value = df.iloc[:, 2].to_numpy(dtype=np.float64)
    else:
        x = df.iloc[:, 0].to_numpy(dtype=np.float64)
        y = df.iloc[:, 1].to_numpy(dtype=np.float64)
        z = df.iloc[:, 2].to_numpy(dtype=np.float64)
        value = df.iloc[:, -1].to_numpy(dtype=np.float64)

        z_round = np.round(z, 6)
        uniq_z = np.unique(z_round)
        z_pick = uniq_z[np.argmin(np.abs(uniq_z - sensor_z_mm))]
        keep = np.isclose(z_round, z_pick, atol=1e-6)
        if keep.sum() == 0:
            raise ValueError(f"{path.name}: cannot find Bz plane near z={sensor_z_mm} mm")
        x = x[keep]
        y = y[keep]
        value = value[keep]

    x_round = np.round(x, 12)
    y_round = np.round(y, 12)
    df2 = pd.DataFrame({"x": x_round, "y": y_round, "v": value})
    df2 = df2.groupby(["y", "x"], as_index=False)["v"].mean()
    pivot = df2.pivot(index="y", columns="x", values="v").sort_index(axis=0).sort_index(axis=1)

    xs = pivot.columns.to_numpy(dtype=np.float64)
    ys = pivot.index.to_numpy(dtype=np.float64)
    grid = pivot.to_numpy(dtype=np.float64)

    qx = np.array([point[0] for point in hall_points_xy], dtype=np.float64)
    qy = np.array([point[1] for point in hall_points_xy], dtype=np.float64)
    if qx.min() < xs.min() or qx.max() > xs.max() or qy.min() < ys.min() or qy.max() > ys.max():
        raise ValueError(
            f"{path.name}: Hall coordinates are outside Bz grid range "
            f"x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}"
        )

    sampled = bilinear_sample_regular_grid(xs, ys, grid, qx, qy).reshape(5, 5)
    return sampled.astype(np.float32)


def load_target_grid(path, target_field, target_size):
    df = pd.read_csv(path)
    expected_cols = {"x_norm", "y_norm", target_field}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{path.name}: expected columns {sorted(expected_cols)}, got {list(df.columns)}")
    pivot = (
        df.pivot(index="y_norm", columns="x_norm", values=target_field)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    grid = pivot.to_numpy(dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D target grid, got shape {grid.shape}")
    if grid.shape != (target_size, target_size):
        grid = resize_map(grid, target_size, mode="bicubic")
    return grid.astype(np.float32)


def cache_file_for_id(cache_dir, sample_id):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_")
    return cache_dir / f"{safe or 'sample'}.npz"


def preprocess_one_sample(
    record,
    cache_dir,
    target_field,
    target_size,
    hall_points_xy,
    sensor_z_mm,
    baseline_strain_label,
    delta_strain_label,
    force=False,
):
    cache_path = cache_file_for_id(cache_dir, record["sample_id"])
    if cache_path.exists() and not force:
        return cache_path

    bz_baseline = load_bz_to_5x5(record["bz"][baseline_strain_label], hall_points_xy, sensor_z_mm)
    bz_loaded = load_bz_to_5x5(record["bz"][delta_strain_label], hall_points_xy, sensor_z_mm)
    # SRCNN input = bicubic-upsampled |delta-Bz| map from the 5x5 Hall grid.
    delta_map = np.abs(bz_loaded - bz_baseline).astype(np.float32)
    input_up = resize_map(delta_map, target_size, mode="bicubic")
    target = load_target_grid(record["target_path"], target_field, target_size)

    np.savez_compressed(
        cache_path,
        sample_id=np.array(record["sample_id"]),
        group=np.array(record["group"]),
        input5=delta_map.astype(np.float32),
        input_up=input_up.astype(np.float32),
        target=target.astype(np.float32),
        target_path=np.array(str(record["target_path"])),
        bz_baseline_path=np.array(str(record["bz"][baseline_strain_label])),
        bz_loaded_path=np.array(str(record["bz"][delta_strain_label])),
        baseline_strain=np.array(baseline_strain_label),
        delta_strain=np.array(delta_strain_label),
    )
    return cache_path


def compute_min_max(cache_paths):
    input_min = float("inf")
    input_max = float("-inf")
    target_min = float("inf")
    target_max = float("-inf")
    # Normalization stats are fit on the training split only to avoid leakage.
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=False) as data:
            input_arr = data["input_up"].astype(np.float32)
            target_arr = data["target"].astype(np.float32)
        input_min = min(input_min, float(input_arr.min()))
        input_max = max(input_max, float(input_arr.max()))
        target_min = min(target_min, float(target_arr.min()))
        target_max = max(target_max, float(target_arr.max()))
    if not math.isfinite(input_min) or not math.isfinite(target_min):
        raise RuntimeError("Failed to compute normalization stats from an empty cache list.")
    return {
        "input_min": input_min,
        "input_max": input_max,
        "target_min": target_min,
        "target_max": target_max,
    }


def normalize_numpy(array, minimum, maximum):
    scale = max(maximum - minimum, EPS)
    return ((array - minimum) / scale).astype(np.float32)


def denormalize_tensor(tensor, minimum, maximum):
    scale = max(maximum - minimum, EPS)
    return tensor * scale + minimum


def stratified_split_ids(sample_ids, seed):
    grouped = {}
    for sample_id in sorted(sample_ids):
        grouped.setdefault(sample_group(sample_id), []).append(sample_id)

    # Keep lesion-source groups separated so train/val/test remain distribution-aware.
    train_ids = []
    val_ids = []
    test_ids = []
    rng = random.Random(seed)

    for group_name in sorted(grouped):
        group_ids = list(grouped[group_name])
        rng.shuffle(group_ids)
        n_total = len(group_ids)
        if n_total == 1:
            train_ids.extend(group_ids)
            continue

        n_test = max(1, int(round(n_total * 0.2)))
        n_test = min(n_test, n_total - 1)
        remaining = n_total - n_test
        n_val = max(1, int(round(remaining * 0.125))) if remaining > 1 else 0
        n_val = min(n_val, max(remaining - 1, 0))

        test_group = group_ids[:n_test]
        val_group = group_ids[n_test : n_test + n_val]
        train_group = group_ids[n_test + n_val :]

        if not train_group and val_group:
            train_group.append(val_group.pop())
        if not train_group and test_group:
            train_group.append(test_group.pop())

        train_ids.extend(sorted(train_group))
        val_ids.extend(sorted(val_group))
        test_ids.extend(sorted(test_group))

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


class CachedPairDataset(Dataset):
    def __init__(self, sample_ids, cache_map, norm_stats):
        self.sample_ids = list(sample_ids)
        self.cache_map = dict(cache_map)
        self.norm_stats = dict(norm_stats)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        cache_path = self.cache_map[sample_id]
        with np.load(cache_path, allow_pickle=False) as data:
            input_up = data["input_up"].astype(np.float32)
            input5 = data["input5"].astype(np.float32)
            target = data["target"].astype(np.float32)
        input_norm = normalize_numpy(input_up, self.norm_stats["input_min"], self.norm_stats["input_max"])
        target_norm = normalize_numpy(target, self.norm_stats["target_min"], self.norm_stats["target_max"])
        return {
            "id": sample_id,
            "input": torch.from_numpy(input_norm).unsqueeze(0).float(),
            "target": torch.from_numpy(target_norm).unsqueeze(0).float(),
            "input_raw": torch.from_numpy(input_up).unsqueeze(0).float(),
            "input_lowres": torch.from_numpy(input5).unsqueeze(0).float(),
            "target_raw": torch.from_numpy(target).unsqueeze(0).float(),
        }


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Classic 3-layer SRCNN backbone: patch extraction, nonlinear mapping, reconstruction.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


def psnr_from_mse(mse):
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=EPS))


def psnr_loss(prediction, target):
    # Training objective: maximize PSNR by minimizing its negative value.
    mse = F.mse_loss(prediction, target)
    return -psnr_from_mse(mse)


def compute_ssim(prediction, target, window_size=7):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padding)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(prediction * prediction, kernel_size=window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size=window_size, stride=1, padding=padding) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / torch.clamp(denominator, min=EPS)).mean()


def compute_metrics(pred_norm, target_norm, target_raw, norm_stats, baseline_norm=None):
    # Report both image-quality metrics (PSNR, SSIM) and raw-scale error metrics (MAE, RMSE).
    pred_raw = denormalize_tensor(pred_norm, norm_stats["target_min"], norm_stats["target_max"])
    pred_mse_norm = F.mse_loss(pred_norm, target_norm)
    pred_psnr = float(psnr_from_mse(pred_mse_norm).item())
    pred_ssim = float(compute_ssim(pred_norm, target_norm).item())
    pred_mae_raw = float(torch.mean(torch.abs(pred_raw - target_raw)).item())
    pred_rmse_raw = float(torch.sqrt(torch.mean((pred_raw - target_raw) ** 2)).item())

    metrics = {
        "pred_psnr": pred_psnr,
        "pred_ssim": pred_ssim,
        "pred_mae_raw": pred_mae_raw,
        "pred_rmse_raw": pred_rmse_raw,
    }

    if baseline_norm is not None:
        # Baseline = range-matched input, used as a simple no-learning reference.
        baseline_raw = denormalize_tensor(baseline_norm, norm_stats["target_min"], norm_stats["target_max"])
        base_mse_norm = F.mse_loss(baseline_norm, target_norm)
        metrics.update(
            {
                "baseline_psnr": float(psnr_from_mse(base_mse_norm).item()),
                "baseline_ssim": float(compute_ssim(baseline_norm, target_norm).item()),
                "baseline_mae_raw": float(torch.mean(torch.abs(baseline_raw - target_raw)).item()),
                "baseline_rmse_raw": float(torch.sqrt(torch.mean((baseline_raw - target_raw) ** 2)).item()),
            }
        )
    return metrics, pred_raw


def save_reconstruction_figure(sample, pred_raw, norm_stats, out_path, target_field, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_norm = sample["input"].cpu()
    input_matched_raw = denormalize_tensor(input_norm, norm_stats["target_min"], norm_stats["target_max"]).squeeze(0).squeeze(0)
    input_lowres = sample["input_lowres"].cpu().squeeze(0).squeeze(0)
    target_raw = sample["target_raw"].cpu().squeeze(0).squeeze(0)
    pred_raw = pred_raw.cpu().squeeze(0).squeeze(0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(title)

    views = [
        (input_lowres.numpy(), "5x5 delta-Bz", "magma"),
        (input_matched_raw.numpy(), "Range-matched input", "viridis"),
        (pred_raw.numpy(), f"SRCNN {target_field}", "viridis"),
        (target_raw.numpy(), f"GT {target_field}", "viridis"),
        (np.abs(pred_raw.numpy() - target_raw.numpy()), "Abs error", "inferno"),
        (np.abs(input_matched_raw.numpy() - target_raw.numpy()), "Input-vs-GT abs error", "inferno"),
    ]

    for axis, (image, subtitle, cmap) in zip(axes.flat, views):
        im = axis.imshow(image, cmap=cmap)
        axis.set_title(subtitle)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_model(model, loader, device, norm_stats, figures_dir=None, target_field="mean_kpa", prefix="eval"):
    model.eval()
    all_rows = []
    aggregate = {}
    for batch_index, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        target_raw = batch["target_raw"].to(device)

        prediction = model(inputs)
        baseline_norm = inputs.clamp(0.0, 1.0)

        # Validation/test use the same metric suite for direct model-vs-baseline comparison.
        metrics, pred_raw = compute_metrics(prediction, targets, target_raw, norm_stats, baseline_norm=baseline_norm)
        sample_id = batch["id"][0]
        row = {"sample_id": sample_id, **metrics}
        all_rows.append(row)

        for key, value in metrics.items():
            aggregate.setdefault(key, []).append(value)

        if figures_dir is not None and batch_index < 4:
            save_reconstruction_figure(
                {key: value[0:1] if torch.is_tensor(value) else value for key, value in batch.items()},
                pred_raw[0:1].cpu(),
                norm_stats,
                figures_dir / f"{prefix}_{batch_index + 1:02d}_{sample_id}.png",
                target_field=target_field,
                title=f"{prefix.upper()} | {sample_id}",
            )

    summary = {key: float(np.mean(values)) for key, values in aggregate.items()} if aggregate else {}
    return summary, all_rows


def build_config(args, hall_points_xy, required_strains):
    return {
        "target_root": str(args.target_root),
        "bz_root": str(args.bz_root),
        "out_dir": str(args.out_dir),
        "target_field": args.target_field,
        "baseline_strain": args.baseline_strain,
        "delta_strain": args.delta_strain,
        "required_strains": list(required_strains),
        "target_size": args.target_size,
        "hall_points_xy_mm": hall_points_xy,
        "sensor_z_mm": args.sensor_z_mm,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
        "max_pairs": args.max_pairs,
        "dry_run": args.dry_run,
        "disable_figures": args.disable_figures,
        "baseline_reference": "range-matched normalized delta-Bz input",
        "srcnn_architecture": [
            "Conv2d(1,64,9x9,pad=4)",
            "Sigmoid",
            "Conv2d(64,32,5x5,pad=2)",
            "Sigmoid",
            "Conv2d(32,1,5x5,pad=2)",
            "Sigmoid",
        ],
    }


def dump_config(path, config, extra=None):
    payload = dict(config)
    if extra:
        payload.update(extra)
    write_json(path, payload)


def build_dataloaders(train_ids, val_ids, test_ids, cache_map, norm_stats, batch_size, num_workers):
    train_ds = CachedPairDataset(train_ids, cache_map, norm_stats)
    val_ds = CachedPairDataset(val_ids, cache_map, norm_stats) if val_ids else None
    test_ds = CachedPairDataset(test_ids, cache_map, norm_stats) if test_ids else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader


def train_model(model, optimizer, train_loader, val_loader, device, norm_stats, args, fixed_batch, figures_dir):
    log_rows = []
    best_val_psnr = float("-inf")
    best_ckpt_path = args.out_dir / "best.pt"
    last_ckpt_path = args.out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            prediction = model(inputs)
            # Optimization is driven by negative PSNR instead of plain MSE.
            loss = psnr_loss(prediction, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_summary = {}
        if val_loader is not None:
            # Select the best checkpoint using validation PSNR.
            val_summary, _ = evaluate_model(
                model,
                val_loader,
                device=device,
                norm_stats=norm_stats,
                figures_dir=None,
                target_field=args.target_field,
                prefix="val",
            )

        fixed_input = fixed_batch["input"].to(device)
        fixed_prediction = model(fixed_input)
        _, fixed_pred_raw = compute_metrics(
            fixed_prediction,
            fixed_batch["target"].to(device),
            fixed_batch["target_raw"].to(device),
            norm_stats,
            baseline_norm=fixed_input,
        )
        if not args.disable_figures:
            save_reconstruction_figure(
                {key: value for key, value in fixed_batch.items()},
                fixed_pred_raw[0:1].cpu(),
                norm_stats,
                figures_dir / f"epoch_{epoch:04d}_{fixed_batch['id'][0]}.png",
                target_field=args.target_field,
                title=f"Epoch {epoch}",
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": round(time.time() - t0, 3),
        }
        row.update({f"val_{key}": value for key, value in val_summary.items()})
        log_rows.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "norm_stats": norm_stats,
            "args": vars(args),
            "val_summary": val_summary,
        }
        torch.save(checkpoint, last_ckpt_path)
        if val_summary.get("pred_psnr", float("-inf")) > best_val_psnr:
            best_val_psnr = val_summary["pred_psnr"]
            torch.save(checkpoint, best_ckpt_path)
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

        print(
            f"[Epoch {epoch:04d}] train_loss={train_loss:.6f}"
            + (
                f" val_psnr={val_summary['pred_psnr']:.4f} val_ssim={val_summary['pred_ssim']:.4f}"
                if val_summary
                else ""
            )
        )

    write_jsonl(args.out_dir / "train_log.jsonl", log_rows)
    return log_rows


def run_training(args):
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cache").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    hall_x = parse_float_list(args.hall_x_mm)
    hall_y = parse_float_list(args.hall_y_mm)
    hall_points_xy = [(x, y) for y in hall_y for x in hall_x]
    baseline_strain_label = strain_label(args.baseline_strain)
    delta_strain_label = strain_label(args.delta_strain)
    required_strains = [baseline_strain_label, delta_strain_label]

    config = build_config(args, hall_points_xy, required_strains)
    dump_config(args.out_dir / "config.json", config)

    if not args.target_root.exists():
        raise FileNotFoundError(f"TARGET_ROOT does not exist: {args.target_root}")
    if not args.bz_root.exists():
        raise FileNotFoundError(f"BZ_ROOT does not exist: {args.bz_root}")

    target_samples = discover_target_samples(args.target_root, args.target_field)
    bz_samples = discover_bz_samples(args.bz_root)
    paired, skipped = pair_samples(target_samples, bz_samples, required_strains)

    if args.max_pairs > 0:
        paired_ids = sorted(paired)[: args.max_pairs]
        paired = {sample_id: paired[sample_id] for sample_id in paired_ids}

    save_pairing_artifacts(
        args.out_dir,
        target_root=args.target_root,
        bz_root=args.bz_root,
        target_field=args.target_field,
        required_strains=required_strains,
        paired=paired,
        skipped=skipped,
    )

    if not paired:
        raise RuntimeError(
            "No paired samples were found. See paired_samples.json for the skip reasons and provide a Bz root "
            "with matching sample IDs plus the required strains."
        )

    cache_dir = args.out_dir / "cache"
    cache_map = {}
    for index, sample_id in enumerate(sorted(paired), start=1):
        cache_map[sample_id] = preprocess_one_sample(
            paired[sample_id],
            cache_dir=cache_dir,
            target_field=args.target_field,
            target_size=args.target_size,
            hall_points_xy=hall_points_xy,
            sensor_z_mm=args.sensor_z_mm,
            baseline_strain_label=baseline_strain_label,
            delta_strain_label=delta_strain_label,
            force=args.force_rebuild_cache,
        )
        if index == 1 or index % 10 == 0 or index == len(paired):
            print(f"Preprocessed {index}/{len(paired)}")

    train_ids, val_ids, test_ids = stratified_split_ids(sorted(cache_map), seed=args.seed)
    if not train_ids:
        raise RuntimeError("Train split is empty after stratified split.")

    # All normalization parameters are derived from the training subset.
    norm_stats = compute_min_max([cache_map[sample_id] for sample_id in train_ids])
    dump_config(
        args.out_dir / "config.json",
        config,
        extra={
            "pair_count": len(cache_map),
            "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
            "norm_stats": norm_stats,
        },
    )

    if args.dry_run:
        print(
            f"Dry run complete. Pairs={len(cache_map)} "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        cache_map=cache_map,
        norm_stats=norm_stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fixed_source = val_loader if val_loader is not None else train_loader
    fixed_batch = next(iter(fixed_source))
    fixed_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in fixed_batch.items()
    }

    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        norm_stats=norm_stats,
        args=args,
        fixed_batch=fixed_batch,
        figures_dir=args.out_dir / "figures",
    )

    best_ckpt_path = args.out_dir / "best.pt"
    if best_ckpt_path.exists():
        # Final test metrics are reported from the best validation checkpoint, not the last epoch.
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    test_summary = {}
    test_rows = []
    if test_loader is not None:
        test_summary, test_rows = evaluate_model(
            model,
            test_loader,
            device=device,
            norm_stats=norm_stats,
            figures_dir=None if args.disable_figures else args.out_dir / "figures",
            target_field=args.target_field,
            prefix="test",
        )

    write_json(
        args.out_dir / "test_metrics.json",
        {
            "summary": test_summary,
            "norm_stats": norm_stats,
            "target_field": args.target_field,
            "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        },
    )
    if test_rows:
        save_csv(args.out_dir / "test_sample_metrics.csv", test_rows)


def main(argv=None):
    args = parse_args(argv)
    try:
        run_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
