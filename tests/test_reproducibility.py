from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from srcnn_8x8_to_32x32.dataset import dataset_summary, load_manifest, split_records
from srcnn_8x8_to_32x32.generate_toy import generate
from srcnn_8x8_to_32x32.infer import load_checkpoint, min_max_normalize, run_inference
from srcnn_8x8_to_32x32.io_utils import load_config, read_grid_csv, write_grid_csv
from srcnn_8x8_to_32x32.metrics import negative_psnr_loss
from srcnn_8x8_to_32x32.model import SRCNN8x8To32x32
from srcnn_8x8_to_32x32.provenance import sha256
from srcnn_8x8_to_32x32.train import compute_norm_stats, run_training

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "srcnn_8x8_to_32x32/config.example.yaml"
torch.set_num_threads(1)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(CONFIG)

    def manifest(self, groups=("a", "b", "c"), splits=("train", "val", "test")):
        rows = []
        for i, (group, split) in enumerate(zip(groups, splits)):
            write_grid_csv(self.root / f"input{i}.csv", np.arange(64).reshape(8, 8) + 100*i)
            write_grid_csv(self.root / f"target{i}.csv", np.arange(1024).reshape(32, 32) + 2000*i)
            rows.append([f"sample{i}", group, split, f"input{i}.csv", f"target{i}.csv"])
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "group_id", "split", "input_8x8_csv", "target_32x32_csv"])
            writer.writerows(rows)
        return path

    def args(self, manifest, out="training", smoke=False):
        return argparse.Namespace(config=CONFIG, manifest=manifest, out_dir=self.root/out,
                                  epochs=1, batch_size=1, lr=1e-4, weight_decay=0.0,
                                  seed=42, device="cpu", smoke_test=smoke)

    def test_model_accepts_only_sensor_grid(self):
        model = SRCNN8x8To32x32()
        self.assertEqual(tuple(model(torch.zeros(2, 1, 8, 8)).shape), (2, 1, 32, 32))
        for shape in ((1, 1, 4, 4), (1, 1, 32, 32), (1, 2, 8, 8)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            SRCNN8x8To32x32(input_size=4)

    def test_csv_does_not_crop_numeric_data_to_fit(self):
        path = self.root / "input.csv"
        for size in (4, 9, 32):
            np.savetxt(path, np.ones((size, size)), delimiter=",")
            with self.assertRaises(ValueError):
                read_grid_csv(path, (8, 8))
        np.savetxt(path, np.ones((8, 8)), delimiter=",")
        self.assertEqual(read_grid_csv(path, (8, 8)).shape, (8, 8))
        bad = np.ones((8, 8)); bad[0, 0] = np.inf
        write_grid_csv(path, bad)
        with self.assertRaises(ValueError):
            read_grid_csv(path, (8, 8))

    def test_group_leakage_rejected(self):
        path = self.manifest(groups=("same", "same", "other"))
        with self.assertRaisesRegex(ValueError, "group_id"):
            load_manifest(path, self.config)

    def test_duplicate_files_across_splits_rejected(self):
        for kind in ("input", "target"):
            path = self.manifest()
            (self.root/f"{kind}1.csv").write_bytes((self.root/f"{kind}0.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, kind + "_sha256"):
                load_manifest(path, self.config)

    def test_explicit_training_and_held_out_splits_required(self):
        path = self.manifest(splits=("val", "val", "test"))
        records = load_manifest(path, self.config)
        with self.assertRaises(ValueError):
            split_records(records, smoke_test=True)
        path = self.manifest(splits=("train", "train", "train"))
        records = load_manifest(path, self.config)
        with self.assertRaises(ValueError):
            split_records(records)
        self.assertEqual(len(split_records(records, smoke_test=True)[0]), 3)

    def test_invalid_manifest_values_rejected(self):
        path = self.manifest()
        original = path.read_text()
        for text in (original.replace("group_id,", "other,"),
                     original.replace("sample1", "sample0"),
                     original.replace("sample0,a,train", "sample0,,train"),
                     original.replace(",val,", ",unknown,")):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_manifest(path, self.config)

    def test_normalization_uses_training_only_without_clipping(self):
        records = load_manifest(self.manifest(), self.config)
        train, _, _ = split_records(records)
        stats = compute_norm_stats(train, 8, 32, "mean_kpa")
        self.assertEqual(stats, {"input_min": 0.0, "input_max": 63.0,
                                 "target_min": 0.0, "target_max": 1023.0})
        self.assertGreater(min_max_normalize(np.array([100.0]), 0, 63)[0], 1)
        self.assertEqual(dataset_summary(records)["independent_group_count"], 3)

    def test_negative_psnr_loss_and_gradient(self):
        prediction = torch.full((2, 1, 8, 8), 0.5, requires_grad=True)
        loss = negative_psnr_loss(prediction, torch.zeros_like(prediction))
        self.assertAlmostEqual(loss.item(), -6.0205999, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad > 0).all())

    def test_missing_checkpoint_cannot_silently_predict(self):
        generate(self.root / "toy")
        with self.assertRaises(FileNotFoundError):
            run_inference(CONFIG, self.root/"toy/toy_input_8x8.csv", self.root/"out", device_name="cpu")
        self.assertFalse((self.root/"out").exists())

    def test_smoke_checkpoint_and_output_provenance(self):
        generate(self.root / "toy")
        args = self.args(self.root/"toy/toy_manifest.csv", smoke=True)
        result = run_training(args)
        self.assertEqual(result["dataset"]["splits"]["test"]["samples"], 0)
        self.assertFalse(result["test_evaluated"])
        self.assertFalse((args.out_dir/"best.pt").exists())
        self.assertFalse((args.out_dir/"test_metrics.json").exists())
        checkpoint = args.out_dir/"last.pt"
        outputs = run_inference(CONFIG, self.root/"toy/toy_input_8x8.csv", self.root/"out", checkpoint, device_name="cpu")
        self.assertEqual(read_grid_csv(outputs["csv"], (32, 32)).shape, (32, 32))
        meta = json.loads(outputs["metadata"].read_text())
        self.assertEqual(meta["run_kind"], "smoke_test")
        self.assertEqual(meta["checkpoint_sha256"], sha256(checkpoint))
        self.assertFalse(meta["manuscript_figure_reproduction"])
        with self.assertRaises(FileExistsError):
            run_training(args)

    def test_grouped_training_selects_validation_then_tests(self):
        args = self.args(self.manifest())
        result = run_training(args)
        self.assertEqual(result["selected_checkpoint"], "best.pt")
        self.assertTrue(result["test_evaluated"])
        metrics = json.loads((args.out_dir/"test_metrics.json").read_text())
        self.assertEqual([r["sample_id"] for r in metrics["samples"]], ["sample2"])
        checkpoint = load_checkpoint(args.out_dir/"best.pt", torch.device("cpu"))
        self.assertEqual(checkpoint["norm_stats"]["input_max"], 63.0)
        self.assertFalse(checkpoint["provenance"]["manuscript_checkpoint"])

    def test_checkpoint_missing_metadata_is_rejected(self):
        path = self.root/"incomplete.pt"
        torch.save({"model_state": SRCNN8x8To32x32().state_dict()}, path)
        with self.assertRaises(ValueError):
            load_checkpoint(path, torch.device("cpu"))

    def test_example_regeneration_is_deterministic(self):
        generate(self.root/"toy")
        paths = list((self.root/"toy").glob("*.csv"))
        before = {p.name: sha256(p) for p in paths}
        generate(self.root/"toy")
        self.assertEqual(before, {p.name: sha256(p) for p in paths})


if __name__ == "__main__":
    unittest.main()
