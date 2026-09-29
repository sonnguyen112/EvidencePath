#!/usr/bin/env python3
"""Prepare a reproducible question sample and its deduplicated corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencepath.datasets import (
    load_corpus,
    load_examples,
    sample_examples,
    write_prepared_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset adapter name.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local JSON or JSONL file.")
    source.add_argument("--hf-dataset", help="Hugging Face dataset repository ID.")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--corpus-input",
        type=Path,
        default=None,
        help="Optional separate corpus JSON/JSONL; otherwise use question-local contexts.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(
        dataset=args.dataset,
        input_path=args.input,
        hf_dataset=args.hf_dataset,
        split=args.split,
        hf_config=args.hf_config,
    )
    examples = sample_examples(examples, sample_size=args.sample_size, seed=args.seed)
    corpus = load_corpus(args.corpus_input) if args.corpus_input else None
    write_prepared_examples(args.output_dir, examples, corpus=corpus)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "seed": args.seed,
            "requested_sample_size": args.sample_size,
            "input": str(args.input) if args.input else None,
            "hf_dataset": args.hf_dataset,
            "split": args.split,
            "corpus_input": str(args.corpus_input) if args.corpus_input else None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
