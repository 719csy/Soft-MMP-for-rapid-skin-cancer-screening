"""Explicit split membership and independent-unit checks for SRCNN datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import resolve_relative
from .provenance import sha256


def load_manifest(path: Path, config: dict) -> list[dict]:
    cfg = config.get("data", {})
    input_column = cfg.get("input_column", "input_8x8_csv")
    target_column = cfg.get("target_column", "target_32x32_csv")
    required = ["sample_id", "group_id", "split", input_column, target_column]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or any(c not in frame for c in required):
        raise ValueError(f"Manifest must be nonempty and contain {required}")
    if any(frame[c].str.strip().eq("").any() for c in required):
        raise ValueError("Manifest identifiers, splits and paths must not be blank")
    if frame["sample_id"].str.strip().duplicated().any():
        raise ValueError("sample_id must be unique")
    records = []
    for _, row in frame.iterrows():
        split = row["split"].strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        input_path = resolve_relative(row[input_column].strip(), path.parent).resolve()
        target_path = resolve_relative(row[target_column].strip(), path.parent).resolve()
        records.append({
            "sample_id": row["sample_id"].strip(), "group_id": row["group_id"].strip(),
            "split": split, "input_path": input_path, "target_path": target_path,
            "input_sha256": sha256(input_path), "target_sha256": sha256(target_path),
        })
    for key in ("group_id", "input_sha256", "target_sha256"):
        seen = {}
        for record in records:
            value = record[key]
            if value in seen and seen[value] != record["split"]:
                raise ValueError(f"Cross-split leakage detected for {key}")
            seen[value] = record["split"]
    return records


def split_records(records: list[dict], smoke_test: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    splits = tuple([r for r in records if r["split"] == name] for name in ("train", "val", "test"))
    if not splits[0]:
        raise ValueError("An explicit training split is required; held-out samples cannot be used for training")
    if not smoke_test and (not splits[1] or not splits[2]):
        raise ValueError("Training requires train, val and test groups; use --smoke_test only for interface checks")
    return splits


def dataset_summary(records: list[dict]) -> dict:
    return {
        "sample_count": len(records),
        "independent_group_count": len({r["group_id"] for r in records}),
        "splits": {
            split: {"samples": sum(r["split"] == split for r in records),
                    "groups": len({r["group_id"] for r in records if r["split"] == split})}
            for split in ("train", "val", "test")
        },
        "samples": [{k: r[k] for k in ("sample_id", "group_id", "split", "input_sha256", "target_sha256")} for r in records],
    }
