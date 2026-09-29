#!/usr/bin/env python3
"""Run hard-cap, ablation, and latency experiments from a prepared index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencepath.datasets import load_prepared_questions
from evidencepath.embeddings import build_encoder
from evidencepath.experiments import ExperimentConfig, ExperimentRunner, write_experiment_results
from evidencepath.generators import build_generator
from evidencepath.retrieval import RetrievalConfig
from evidencepath.text import HuggingFaceTokenizer, SimpleTokenizer
from evidencepath.types import EvidenceGraph


DEFAULT_METHODS = ["evidencepath", "hipporag", "sentence", "linear", "raptor", "lightrag", "vanilla"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1536, 2048, 3072])
    parser.add_argument("--ablations", nargs="+", default=["full"])
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--warmup-questions",
        type=int,
        default=50,
        help="Number of initial questions used for unmeasured warmup; set 0 to disable.",
    )
    parser.add_argument("--encoder-backend", choices=["sentence-transformers", "huggingface", "hashing"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hashing-dimension", type=int, default=384)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--quantize-8bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-policy", choices=["hardcap", "original"], default="hardcap")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--terminal-multiplier", type=float, default=0.30)
    parser.add_argument("--edge-cutoff", type=float, default=0.05)
    parser.add_argument("--augmentation-cutoff", type=float, default=0.60)
    parser.add_argument("--hard-cap-target-fraction", type=float, default=0.90)
    parser.add_argument("--no-cuda-sync", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = EvidenceGraph.from_files(args.index)
    questions = load_prepared_questions(args.questions)
    if args.max_questions is not None:
        questions = questions[: max(0, args.max_questions)]

    index_metadata = graph.metadata
    encoder_backend = args.encoder_backend or str(index_metadata.get("encoder_backend", "sentence-transformers"))
    embedding_model = args.embedding_model or str(
        index_metadata.get("encoder_model", "BAAI/bge-m3")
    )
    tokenizer_model = args.tokenizer_model
    if tokenizer_model:
        tokenizer = HuggingFaceTokenizer(tokenizer_model)
    else:
        tokenizer = SimpleTokenizer()
    encoder = build_encoder(
        encoder_backend,
        embedding_model,
        batch_size=32,
        device=args.device,
        max_seq_length=512,
        hashing_dimension=args.hashing_dimension,
    )
    generator = build_generator(
        args.generator_model,
        quantize_8bit=args.quantize_8bit,
        max_input_tokens=16_384,
        max_answer_tokens=64,
    )
    retrieval_config = RetrievalConfig(
        top_k=args.top_k,
        alpha=args.alpha,
        iterations=args.iterations,
        threshold=args.threshold,
        terminal_multiplier=args.terminal_multiplier,
        edge_cutoff=args.edge_cutoff,
        augmentation_cutoff=args.augmentation_cutoff,
        hard_cap_target_fraction=args.hard_cap_target_fraction,
    )
    runner = ExperimentRunner(
        graph,
        encoder,
        tokenizer,
        generator,
        ExperimentConfig(
            retrieval=retrieval_config,
            synchronize_cuda=not args.no_cuda_sync,
            hard_cap=args.routing_policy == "hardcap",
        ),
    )
    runner.warmup(
        questions,
        methods=args.methods,
        budgets=args.budgets,
        ablations=args.ablations,
        count=args.warmup_questions,
    )
    rows = runner.run(
        questions,
        methods=args.methods,
        budgets=args.budgets,
        ablations=args.ablations,
    )
    metadata = {
        "index": str(args.index),
        "questions": str(args.questions),
        "encoder_backend": encoder_backend,
        "embedding_model": embedding_model,
        "tokenizer_model": tokenizer_model,
        "generator_model": args.generator_model,
        "methods": args.methods,
        "budgets": args.budgets,
        "ablations": args.ablations,
        "routing_policy": args.routing_policy,
        "num_questions": len(questions),
        "retrieval_config": vars(retrieval_config),
    }
    write_experiment_results(args.output_dir, rows, metadata=metadata)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
