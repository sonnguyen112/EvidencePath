"""Self-contained baselines used by the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from .embeddings import EmbeddingEncoder, l2_normalize
from .packing import ContextPacker, generic_blocks_from_chunks, generic_blocks_from_sentences
from .text import TokenizerProtocol
from .types import EvidenceGraph, PackedContext


@dataclass
class BaselineResult:
    context: PackedContext
    preassembly_source_ids: tuple[str, ...]
    metadata: dict[str, object]


class _BaseBaseline:
    def __init__(self, graph: EvidenceGraph, encoder: EmbeddingEncoder, tokenizer: TokenizerProtocol) -> None:
        self.graph = graph
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.packer = ContextPacker(tokenizer, graph)

    def _pack_chunks(
        self,
        query: str,
        scores: Mapping[str, float],
        *,
        cap: int,
        chunk_ids: list[str] | None = None,
        method: str,
    ) -> BaselineResult:
        selected = (
            sorted(self.graph.chunks, key=lambda item: (-scores[item], item))
            if chunk_ids is None
            else chunk_ids
        )
        blocks = generic_blocks_from_chunks(self.graph, selected, scores)
        context = self.packer.pack_blocks(blocks, cap=cap, order="score")
        preassembly = tuple(
            dict.fromkeys(
                source_id
                for chunk_id in selected
                for source_id in self.graph.chunks[chunk_id].sentence_unit_ids
            )
        )
        return BaselineResult(
            context=context,
            preassembly_source_ids=preassembly,
            metadata={"method": method, "candidate_chunks": len(selected)},
        )

    def _chunk_scores(self, query: str) -> dict[str, float]:
        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        return {
            chunk_id: float(np.dot(query_vector, self.graph.chunk_embeddings[chunk_id]))
            for chunk_id in self.graph.chunks
        }


class SentenceRetrieval(_BaseBaseline):
    """Independent sentence retrieval with the common greedy-fit rule."""

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        scores = {
            source_id: float(np.dot(query_vector, embedding))
            for source_id, embedding in self.graph.source_embeddings.items()
        }
        selected = sorted(scores, key=lambda item: (-scores[item], item))
        context = self.packer.pack_blocks(
            generic_blocks_from_sentences(self.graph, selected, scores),
            cap=cap,
            order="score",
        )
        return BaselineResult(
            context=context,
            preassembly_source_ids=tuple(selected),
            metadata={"method": "sentence", "candidate_sentences": len(selected)},
        )


class VanillaRAG(_BaseBaseline):
    """Chunk-level retrieval without a separate sentence-selection stage."""

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        return self._pack_chunks(query, self._chunk_scores(query), cap=cap, method="vanilla")


class HippoRAGStyle(_BaseBaseline):
    """A dependency-free PPR graph baseline following HippoRAG's routing idea."""

    def __init__(
        self,
        graph: EvidenceGraph,
        encoder: EmbeddingEncoder,
        tokenizer: TokenizerProtocol,
        *,
        top_k: int = 20,
        damping: float = 0.85,
        iterations: int = 8,
    ) -> None:
        super().__init__(graph, encoder, tokenizer)
        self.top_k = top_k
        self.damping = damping
        self.iterations = iterations

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        chunk_scores = {
            chunk_id: float(np.dot(query_vector, self.graph.chunk_embeddings[chunk_id]))
            for chunk_id in self.graph.chunks
        }
        nodes = sorted(self.graph.chunks)
        seed_nodes = sorted(nodes, key=lambda node: (-chunk_scores[node], node))[: self.top_k]
        activation = {node: 0.0 for node in nodes}
        if seed_nodes:
            maximum = max(chunk_scores[node] for node in seed_nodes)
            for node in seed_nodes:
                activation[node] = max(0.0, chunk_scores[node] - maximum + 1.0)
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in self.graph.edges.values():
            if edge.embedding is None:
                continue
            edge_score = max(0.0, float(np.dot(query_vector, edge.embedding)))
            if edge_score <= 0.0:
                continue
            adjacency[edge.u].append((edge.v, edge_score))
            adjacency[edge.v].append((edge.u, edge_score))
        for node in adjacency:
            adjacency[node].sort(key=lambda value: value[0])
        for _ in range(max(0, self.iterations)):
            updated = {node: (1.0 - self.damping) * activation[node] for node in nodes}
            for node in nodes:
                neighbors = adjacency.get(node, ())
                normalizer = sum(weight for _, weight in neighbors)
                if normalizer <= 0.0:
                    continue
                for neighbor, weight in neighbors:
                    updated[neighbor] += self.damping * activation[node] * weight / normalizer
            activation = updated
        combined = {
            node: 0.5 * activation[node] + 0.5 * max(0.0, chunk_scores[node])
            for node in nodes
        }
        selected = sorted(nodes, key=lambda node: (-combined[node], node))
        result = self._pack_chunks(
            query,
            combined,
            cap=cap,
            chunk_ids=selected,
            method="hipporag",
        )
        result.metadata.update({"ppr_iterations": self.iterations, "damping": self.damping})
        return result


class LinearRAG(_BaseBaseline):
    """Linear graph baseline that expands from high-scoring chunks."""

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        scores = self._chunk_scores(query)
        ordered = sorted(scores, key=lambda item: (-scores[item], item))
        expanded: list[str] = []
        seen: set[str] = set()
        adjacency = self.graph.adjacency()
        for seed in ordered:
            if seed not in seen:
                expanded.append(seed)
                seen.add(seed)
            neighbors = sorted(
                {
                    edge.v if edge.u == seed else edge.u
                    for edge in adjacency.get(seed, ())
                },
                key=lambda node: (-scores.get(node, 0.0), node),
            )
            for neighbor in neighbors[:2]:
                if neighbor not in seen:
                    expanded.append(neighbor)
                    seen.add(neighbor)
        return self._pack_chunks(query, scores, cap=cap, chunk_ids=expanded, method="linear")


class RaptorStyle(_BaseBaseline):
    """Hierarchical document-centroid retrieval without generated summaries."""

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        by_document: dict[str, list[str]] = defaultdict(list)
        for chunk_id, chunk in self.graph.chunks.items():
            by_document[chunk.doc_id].append(chunk_id)
        document_scores: dict[str, float] = {}
        for doc_id, chunk_ids in by_document.items():
            centroid = l2_normalize(
                np.mean(np.stack([self.graph.chunk_embeddings[item] for item in chunk_ids]), axis=0)
            )[0]
            document_scores[doc_id] = float(np.dot(query_vector, centroid))
        chunk_scores = {
            chunk_id: 0.5 * document_scores[chunk.doc_id]
            + 0.5 * float(np.dot(query_vector, self.graph.chunk_embeddings[chunk_id]))
            for chunk_id, chunk in self.graph.chunks.items()
        }
        selected = sorted(chunk_scores, key=lambda item: (-chunk_scores[item], item))
        result = self._pack_chunks(
            query,
            chunk_scores,
            cap=cap,
            chunk_ids=selected,
            method="raptor",
        )
        result.metadata["hierarchy_documents"] = len(by_document)
        return result


class LightRAGStyle(_BaseBaseline):
    """Mixed low-level chunk and high-level relational retrieval baseline."""

    def retrieve(self, query: str, *, cap: int) -> BaselineResult:
        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        direct = {
            chunk_id: float(np.dot(query_vector, self.graph.chunk_embeddings[chunk_id]))
            for chunk_id in self.graph.chunks
        }
        relational: dict[str, float] = defaultdict(float)
        for edge in self.graph.edges.values():
            if edge.embedding is None:
                continue
            score = max(0.0, float(np.dot(query_vector, edge.embedding)))
            relational[edge.u] = max(relational[edge.u], score)
            relational[edge.v] = max(relational[edge.v], score)
        scores = {
            chunk_id: 0.5 * max(0.0, direct[chunk_id]) + 0.5 * relational[chunk_id]
            for chunk_id in self.graph.chunks
        }
        selected = sorted(scores, key=lambda item: (-scores[item], item))
        return self._pack_chunks(query, scores, cap=cap, chunk_ids=selected, method="lightrag")


def build_baseline(
    name: str,
    graph: EvidenceGraph,
    encoder: EmbeddingEncoder,
    tokenizer: TokenizerProtocol,
    *,
    top_k: int = 20,
) -> object:
    """Create a named baseline retriever."""

    normalized = name.casefold().replace("_", "-")
    if normalized in {"sentence", "sentence-retrieval", "sr"}:
        return SentenceRetrieval(graph, encoder, tokenizer)
    if normalized in {"vanilla", "vanilla-rag"}:
        return VanillaRAG(graph, encoder, tokenizer)
    if normalized in {"hipporag", "hippo-rag"}:
        return HippoRAGStyle(graph, encoder, tokenizer, top_k=top_k)
    if normalized == "linear":
        return LinearRAG(graph, encoder, tokenizer)
    if normalized == "raptor":
        return RaptorStyle(graph, encoder, tokenizer)
    if normalized in {"lightrag", "light-rag"}:
        return LightRAGStyle(graph, encoder, tokenizer)
    raise ValueError(f"Unknown baseline method: {name}")
