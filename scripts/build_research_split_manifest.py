#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and freeze the train_fit/train_tune split manifest used by comparable baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oglans.config import ConfigManager
from oglans.data.adapter import DuEEFinAdapter
from oglans.utils.research_protocol import (
    DEFAULT_TRAIN_TUNE_RATIO,
    build_research_split_manifest,
    save_research_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen research split manifest.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/raw/DuEE-Fin")
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tune_ratio", type=float, default=DEFAULT_TRAIN_TUNE_RATIO)
    parser.add_argument(
        "--output",
        type=str,
        default="configs/research_splits/duee_fin_train_seed3407_tune0.1.json",
    )
    args = parser.parse_args()

    config = ConfigManager.load_config(args.config)
    taxonomy_path = Path(
        config.get("algorithms", {})
        .get("ds_cns", {})
        .get("taxonomy_path", f"{args.data_dir}/duee_fin_event_schema.json")
    )
    if not taxonomy_path.is_absolute():
        taxonomy_path = (Path.cwd() / taxonomy_path).resolve()
    if not taxonomy_path.exists():
        taxonomy_path = Path(args.data_dir) / "duee_fin_event_schema.json"
    adapter = DuEEFinAdapter(data_path=args.data_dir, schema_path=str(taxonomy_path))
    samples = adapter.load_data(args.split)
    manifest = build_research_split_manifest(
        samples,
        tune_ratio=args.tune_ratio,
        seed=args.seed,
    )
    manifest["dataset"] = "DuEE-Fin"
    manifest["source_split"] = args.split
    output_path = save_research_split_manifest(manifest, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
