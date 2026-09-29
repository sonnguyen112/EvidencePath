#!/usr/bin/env python3
"""Build and serialize the reusable EvidencePath evidence graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

from evidencepath.datasets import load_corpus
from evidencepath.embeddings import build_encoder
from evidencepath.entities import build_entity_extractor
from evidencepath.index import EvidenceGraphBuilder, IndexConfig
from evidencepath.text import HuggingFaceTokenizer, SimpleTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-backend", choices=["sentence-transformers", "huggingface", "hashing"], default="sentence-transformers")
    parser.add_argument("--encoder-model", default="BAAI/bge-m3")
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--entity-backend", choices=["spacy", "regex", "spacy+regex"], default="spacy")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--embedding-max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-entity", type=float, default=0.40)
    parser.add_argument("--semantic-neighbors", type=int, default=1)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--hashing-dimension", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    documents = load_corpus(args.corpus)
    if args.tokenizer_model:
        tokenizer = HuggingFaceTokenizer(args.tokenizer_model)
    else:
        tokenizer = SimpleTokenizer()
    encoder = build_encoder(
        args.encoder_backend,
        args.encoder_model,
        batch_size=args.batch_size,
        device=args.device,
        max_seq_length=args.embedding_max_tokens,
        hashing_dimension=args.hashing_dimension,
    )
    extractor = build_entity_extractor(args.entity_backend, spacy_model=args.spacy_model)
    config = IndexConfig(
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        embedding_max_tokens=args.embedding_max_tokens,
        embedding_batch_size=args.batch_size,
        lambda_entity=args.lambda_entity,
        semantic_neighbors=args.semantic_neighbors,
        deduplicate_documents=not args.keep_duplicates,
    )
    graph = EvidenceGraphBuilder(encoder, extractor, tokenizer, config).build(documents)
    graph.metadata.update(
        {
            "build_seconds_before_serialization": time.perf_counter() - start_time,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
    )
    graph.metadata.update(
        {
            "encoder_backend": args.encoder_backend,
            "encoder_model": args.encoder_model,
            "tokenizer_model": args.tokenizer_model,
            "entity_backend": args.entity_backend,
        }
    )
    graph.to_files(args.output)
    index_size_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    graph.metadata["index_size_bytes"] = index_size_bytes
    graph.metadata["build_seconds"] = time.perf_counter() - start_time
    graph.to_files(args.output)
    (args.output / "build_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
