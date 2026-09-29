"""Experiment orchestration shared by the command-line scripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from collections.abc import Iterable, Sequence

from .baselines import BaselineResult, build_baseline
from .datasets import write_jsonl
from .generators import AnswerGenerator
from .metrics import (
    aggregate_prediction_rows,
    answer_exact_match,
    answer_f1,
    coverage_for_question,
    preassembly_units,
)
from .retrieval import EvidencePathRetriever, RetrievalConfig
from .text import TokenizerProtocol
from .types import EvidenceGraph, QuestionExample, RetrievalResult
from .embeddings import EmbeddingEncoder


@dataclass(frozen=True)
class ExperimentConfig:
    retrieval: RetrievalConfig = RetrievalConfig()
    synchronize_cuda: bool = True
    hard_cap: bool = True


def _cuda_synchronize(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _peak_gpu_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**3))
    except ImportError:
        pass
    return 0.0


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


class ExperimentRunner:
    """Run identical questions through methods and context budgets."""

    def __init__(
        self,
        graph: EvidenceGraph,
        encoder: EmbeddingEncoder,
        tokenizer: TokenizerProtocol,
        generator: AnswerGenerator,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.graph = graph
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.generator = generator
        self.config = config or ExperimentConfig()

    def _evidencepath(
        self,
        *,
        ablation: str,
    ) -> EvidencePathRetriever:
        edge_filter = None
        if ablation == "no_semantic_bridge":
            edge_filter = lambda edge: edge.relation_type != "semantic_bridge"
        return EvidencePathRetriever(
            self.graph,
            self.encoder,
            self.tokenizer,
            self.config.retrieval,
            edge_filter=edge_filter,
        )

    def _retrieve(
        self,
        method: str,
        query: str,
        *,
        budget: int,
        ablation: str,
    ) -> tuple[object, tuple[str, ...], dict[str, object]]:
        normalized = method.casefold().replace("_", "-")
        if normalized in {"evidencepath", "evidencepath-rag", "epr"}:
            retriever = self._evidencepath(ablation=ablation)
            retrieval = retriever.retrieve(
                query,
                context_cap=budget,
                hard_cap=self.config.hard_cap,
                solver="greedy" if ablation == "no_steiner" else "kmb",
                disable_propagation=ablation == "no_propagation",
                pack_extra_candidates=ablation in {"no_propagation", "no_semantic_bridge"},
            )
            return retrieval.context, retrieval.selected_source_unit_ids, {
                "retrieval": retrieval,
                "selected_edge_ids": list(retrieval.selected_edge_ids),
                "preassembly_source_ids": list(retrieval.selected_source_unit_ids),
            }
        if ablation != "full":
            raise ValueError(f"Ablation {ablation!r} is only valid for evidencepath")
        baseline = build_baseline(
            method,
            self.graph,
            self.encoder,
            self.tokenizer,
            top_k=self.config.retrieval.top_k,
        )
        result: BaselineResult = baseline.retrieve(query, cap=budget)  # type: ignore[attr-defined]
        return result.context, result.preassembly_source_ids, {
            "baseline": result,
            "preassembly_source_ids": list(result.preassembly_source_ids),
        }

    def run(
        self,
        questions: Sequence[QuestionExample],
        *,
        methods: Sequence[str],
        budgets: Sequence[int],
        ablations: Sequence[str] = ("full",),
    ) -> list[dict[str, object]]:
        """Return one JSON-serializable record per question/method/budget."""

        rows: list[dict[str, object]] = []
        for method in methods:
            for ablation in ablations:
                if method.casefold() not in {"evidencepath", "evidencepath-rag", "epr"} and ablation != "full":
                    continue
                for budget in budgets:
                    for example in questions:
                        _cuda_synchronize(self.config.synchronize_cuda)
                        retrieval_start = time.perf_counter()
                        context, preassembly_source_ids, details = self._retrieve(
                            method,
                            example.question,
                            budget=budget,
                            ablation=ablation,
                        )
                        _cuda_synchronize(self.config.synchronize_cuda)
                        retrieval_seconds = time.perf_counter() - retrieval_start

                        generation_start = time.perf_counter()
                        prediction = self.generator.answer(example.question, context.text)
                        _cuda_synchronize(self.config.synchronize_cuda)
                        generation_seconds = time.perf_counter() - generation_start

                        post_support, post_complete = coverage_for_question(
                            example,
                            retained_sentence_ids=context.source_unit_ids,
                            retained_paragraph_ids=context.paragraph_unit_ids,
                        )
                        pre_units = preassembly_units(
                            preassembly_source_ids,
                            self.graph.source_units,
                            support_level=example.support_level,
                        )
                        pre_support, _ = (
                            coverage_for_question(
                                example,
                                retained_sentence_ids=preassembly_source_ids,
                                retained_paragraph_ids=pre_units,
                            )
                        )
                        row: dict[str, object] = {
                            "question_id": example.question_id,
                            "dataset": example.dataset,
                            "method": method,
                            "ablation": ablation,
                            "budget": int(budget),
                            "question": example.question,
                            "prediction": prediction,
                            "references": list(example.answers),
                            "em": answer_exact_match(prediction, example.answers),
                            "f1": answer_f1(prediction, example.answers),
                            "context_tokens": int(context.token_count),
                            "support_recall": post_support,
                            "complete_coverage": post_complete,
                            "pre_support_recall": pre_support,
                            "context_edge_ids": list(context.edge_ids),
                            "context_source_unit_ids": list(context.source_unit_ids),
                            "retrieval_seconds": retrieval_seconds,
                            "generation_seconds": generation_seconds,
                            "end_to_end_seconds": retrieval_seconds + generation_seconds,
                            "peak_gpu_memory_gb": _peak_gpu_memory_gb(),
                        }
                        retrieval_details = details.get("retrieval")
                        if isinstance(retrieval_details, RetrievalResult):
                            selected_vertices = set(
                                retrieval_details.metadata.get("selected_vertices", [])
                            )
                            terminal_nodes = set(retrieval_details.terminal_nodes)
                            terminal_coverage = (
                                len(selected_vertices & terminal_nodes) / len(terminal_nodes)
                                if terminal_nodes
                                else 0.0
                            )
                            row.update(
                                {
                                    "effective_threshold": retrieval_details.effective_threshold,
                                    "terminal_nodes": list(retrieval_details.terminal_nodes),
                                    "candidate_nodes": list(retrieval_details.candidate_nodes),
                                    "selected_edge_ids": list(retrieval_details.selected_edge_ids),
                                    "selected_source_unit_ids": list(
                                        retrieval_details.selected_source_unit_ids
                                    ),
                                    "candidate_edges": retrieval_details.metadata.get(
                                        "candidate_edges", 0
                                    ),
                                    "selected_vertices": retrieval_details.metadata.get(
                                        "selected_vertices", []
                                    ),
                                    "terminal_coverage": terminal_coverage,
                                    "selected_intermediates": float(
                                        bool(selected_vertices - terminal_nodes)
                                    ),
                                }
                            )
                        rows.append(row)
        return rows

    def warmup(
        self,
        questions: Sequence[QuestionExample],
        *,
        methods: Sequence[str],
        budgets: Sequence[int],
        ablations: Sequence[str] = ("full",),
        count: int = 50,
    ) -> None:
        """Warm up retrieval and generation without recording measurements."""

        if count <= 0:
            return
        warmup_questions = questions[:count]
        warmup_budget = budgets[0] if budgets else 2048
        for method in methods:
            for ablation in ablations:
                if method.casefold() not in {"evidencepath", "evidencepath-rag", "epr"} and ablation != "full":
                    continue
                for example in warmup_questions:
                    context, _, _ = self._retrieve(
                        method,
                        example.question,
                        budget=warmup_budget,
                        ablation=ablation,
                    )
                    self.generator.answer(example.question, context.text)
                    _cuda_synchronize(self.config.synchronize_cuda)


def write_experiment_results(
    output_dir: str | Path,
    rows: Sequence[dict[str, object]],
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write per-query files and an aggregate summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            str(row["ablation"]),
            int(row["budget"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (dataset, method, ablation, budget), group in sorted(groups.items()):
        filename = (
            f"{_safe_name(dataset)}__{_safe_name(method)}__{_safe_name(ablation)}__{budget}.jsonl"
        )
        write_jsonl(output / filename, group)
        summary = {
            "dataset": dataset,
            "method": method,
            "ablation": ablation,
            "budget": budget,
            **aggregate_prediction_rows(group),
        }
        summaries.append(summary)
    payload = {"metadata": metadata or {}, "summaries": summaries}
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
