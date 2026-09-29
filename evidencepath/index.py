"""Offline construction of the sentence-bearing EvidencePath graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from itertools import combinations
import hashlib
import math
from collections.abc import Iterable, Sequence

import numpy as np

from .embeddings import EmbeddingEncoder, l2_normalize
from .entities import EntityExtractor
from .text import TokenizerProtocol, deduplicate_documents, normalize_entity, sentence_aware_chunks
from .types import Chunk, DocumentRecord, EvidenceEdge, EvidenceGraph, SentenceUnit


@dataclass(frozen=True)
class IndexConfig:
    chunk_tokens: int = 256
    chunk_overlap: int = 64
    embedding_max_tokens: int = 512
    embedding_batch_size: int = 32
    lambda_entity: float = 0.40
    semantic_neighbors: int = 1
    deduplicate_documents: bool = True


def _stable_digest(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


class EvidenceGraphBuilder:
    """Build a reusable graph whose edges retain source sentences."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        entity_extractor: EntityExtractor,
        tokenizer: TokenizerProtocol,
        config: IndexConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.entity_extractor = entity_extractor
        self.tokenizer = tokenizer
        self.config = config or IndexConfig()

    def build(self, documents: Iterable[DocumentRecord]) -> EvidenceGraph:
        documents_list = list(documents)
        if self.config.deduplicate_documents:
            documents_list = deduplicate_documents(documents_list)

        chunks: dict[str, Chunk] = {}
        source_units: dict[str, SentenceUnit] = {}
        units_by_chunk: dict[str, tuple[SentenceUnit, ...]] = {}
        for document in documents_list:
            document_chunks, units = sentence_aware_chunks(
                document,
                self.tokenizer,
                max_tokens=self.config.chunk_tokens,
                overlap_tokens=self.config.chunk_overlap,
            )
            for unit in units:
                source_units[unit.unit_id] = unit
            unit_by_id = {unit.unit_id: unit for unit in units}
            for chunk in document_chunks:
                chunks[chunk.chunk_id] = chunk
                units_by_chunk[chunk.chunk_id] = tuple(
                    unit_by_id[unit_id] for unit_id in chunk.sentence_unit_ids
                )

        if not chunks:
            return EvidenceGraph(
                chunks={},
                edges={},
                source_units={},
                chunk_embeddings={},
                source_embeddings={},
                metadata={"config": asdict(self.config), "num_documents": len(documents_list)},
            )

        ordered_chunk_ids = sorted(chunks)
        chunk_vectors = l2_normalize(self.encoder.encode([chunks[item].text for item in ordered_chunk_ids]))
        chunk_embeddings = {
            chunk_id: chunk_vectors[index] for index, chunk_id in enumerate(ordered_chunk_ids)
        }

        ordered_source_ids = sorted(source_units)
        source_vectors = l2_normalize(
            self.encoder.encode([source_units[item].text for item in ordered_source_ids])
        )
        source_embeddings = {
            source_id: source_vectors[index] for index, source_id in enumerate(ordered_source_ids)
        }

        entity_sets, entity_tf = self._extract_entities(chunks)
        for chunk_id, values in entity_sets.items():
            chunks[chunk_id].entities = tuple(sorted(values))
        selected_entities = self._select_entities(entity_sets, entity_tf)

        edges: dict[str, EvidenceEdge] = {}
        self._add_entity_edges(
            edges,
            chunks,
            units_by_chunk,
            selected_entities,
            source_embeddings,
        )
        self._add_semantic_edges(edges, chunks, units_by_chunk, chunk_embeddings, source_embeddings)

        edge_ids = sorted(edges)
        edge_texts = [edges[edge_id].evidence_text for edge_id in edge_ids]
        edge_vectors = l2_normalize(self.encoder.encode(edge_texts)) if edge_texts else np.empty((0, 0))
        for index, edge_id in enumerate(edge_ids):
            edges[edge_id].embedding = edge_vectors[index]

        metadata = {
            "config": asdict(self.config),
            "num_documents": len(documents_list),
            "num_chunks": len(chunks),
            "num_source_units": len(source_units),
            "num_edges": len(edges),
            "entity_edges": sum(edge.relation_type == "entity_sharing" for edge in edges.values()),
            "semantic_edges": sum(edge.relation_type == "semantic_bridge" for edge in edges.values()),
        }
        return EvidenceGraph(
            chunks=chunks,
            edges=edges,
            source_units=source_units,
            chunk_embeddings=chunk_embeddings,
            source_embeddings=source_embeddings,
            metadata=metadata,
        )

    def _extract_entities(
        self, chunks: dict[str, Chunk]
    ) -> tuple[dict[str, set[str]], dict[str, Counter[str]]]:
        entity_sets: dict[str, set[str]] = {}
        entity_tf: dict[str, Counter[str]] = {}
        for chunk_id in sorted(chunks):
            extracted = [normalize_entity(value) for value in self.entity_extractor.extract(chunks[chunk_id].text)]
            extracted = [value for value in extracted if value]
            counts: Counter[str] = Counter()
            for entity in extracted:
                occurrences = chunks[chunk_id].text.casefold().count(entity.casefold())
                counts[entity] += max(1, occurrences)
            entity_sets[chunk_id] = set(counts)
            entity_tf[chunk_id] = counts
        return entity_sets, entity_tf

    def _select_entities(
        self,
        entity_sets: dict[str, set[str]],
        entity_tf: dict[str, Counter[str]],
    ) -> dict[str, set[str]]:
        document_frequency: Counter[str] = Counter()
        for values in entity_sets.values():
            document_frequency.update(values)
        number_of_chunks = max(1, len(entity_sets))
        selected: dict[str, set[str]] = {}
        for chunk_id in sorted(entity_sets):
            weights: dict[str, float] = {}
            for entity in entity_sets[chunk_id]:
                tf = entity_tf[chunk_id][entity]
                df = max(1, document_frequency[entity])
                weights[entity] = float(tf) * math.log(number_of_chunks / df)
            maximum = max(weights.values(), default=0.0)
            selected[chunk_id] = {
                entity
                for entity, weight in weights.items()
                if weight > self.config.lambda_entity * maximum
            }
        return selected

    @staticmethod
    def _best_entity_sentence(
        entity: str,
        units: Sequence[SentenceUnit],
        source_embeddings: dict[str, np.ndarray],
    ) -> SentenceUnit:
        normalized_entity = normalize_entity(entity)
        matches = [
            index
            for index, unit in enumerate(units)
            if normalized_entity in normalize_entity(unit.text)
        ]
        if not matches:
            return min(units, key=lambda unit: (unit.sentence_id, unit.unit_id))
        if len(matches) == 1:
            return units[matches[0]]
        best_index = matches[0]
        best_score = -float("inf")
        for index in matches:
            local_indices = range(max(0, index - 1), min(len(units), index + 2))
            local_vectors = [source_embeddings[units[item].unit_id] for item in local_indices]
            local_vector = l2_normalize(np.mean(np.stack(local_vectors), axis=0))[0]
            score = float(np.dot(source_embeddings[units[index].unit_id], local_vector))
            candidate = units[index]
            if score > best_score or (
                score == best_score and (candidate.sentence_id, candidate.unit_id)
                < (units[best_index].sentence_id, units[best_index].unit_id)
            ):
                best_index = index
                best_score = score
        return units[best_index]

    def _add_entity_edges(
        self,
        edges: dict[str, EvidenceEdge],
        chunks: dict[str, Chunk],
        units_by_chunk: dict[str, tuple[SentenceUnit, ...]],
        selected_entities: dict[str, set[str]],
        source_embeddings: dict[str, np.ndarray],
    ) -> None:
        chunks_by_entity: dict[str, list[str]] = defaultdict(list)
        for chunk_id, entities in selected_entities.items():
            for entity in entities:
                chunks_by_entity[entity].append(chunk_id)
        for entity in sorted(chunks_by_entity):
            chunk_ids = sorted(chunks_by_entity[entity])
            for left, right in combinations(chunk_ids, 2):
                left_sentence = self._best_entity_sentence(
                    entity, units_by_chunk[left], source_embeddings
                )
                right_sentence = self._best_entity_sentence(
                    entity, units_by_chunk[right], source_embeddings
                )
                edge_id = f"entity-{_stable_digest(left, right, entity)}"
                edges[edge_id] = EvidenceEdge(
                    edge_id=edge_id,
                    u=min(left, right),
                    v=max(left, right),
                    relation_type="entity_sharing",
                    relation_key=entity,
                    evidence=(left_sentence, right_sentence),
                    metadata={"entity": entity},
                )

    @staticmethod
    def _best_sentence_pair(
        left_units: Sequence[SentenceUnit],
        right_units: Sequence[SentenceUnit],
        source_embeddings: dict[str, np.ndarray],
    ) -> tuple[SentenceUnit, SentenceUnit]:
        best: tuple[float, tuple[int, str], tuple[int, str], SentenceUnit, SentenceUnit] | None = None
        for left in left_units:
            for right in right_units:
                score = float(
                    np.dot(source_embeddings[left.unit_id], source_embeddings[right.unit_id])
                )
                candidate = (
                    score,
                    (left.sentence_id, left.unit_id),
                    (right.sentence_id, right.unit_id),
                    left,
                    right,
                )
                if best is None or score > best[0] or (
                    score == best[0] and candidate[1:3] < best[1:3]
                ):
                    best = candidate
        if best is None:
            raise ValueError("Semantic edges require both chunks to contain source sentences")
        return best[3], best[4]

    def _add_semantic_edges(
        self,
        edges: dict[str, EvidenceEdge],
        chunks: dict[str, Chunk],
        units_by_chunk: dict[str, tuple[SentenceUnit, ...]],
        chunk_embeddings: dict[str, np.ndarray],
        source_embeddings: dict[str, np.ndarray],
    ) -> None:
        if self.config.semantic_neighbors <= 0 or len(chunks) < 2:
            return
        chunk_ids = sorted(chunks)
        matrix = np.stack([chunk_embeddings[chunk_id] for chunk_id in chunk_ids])
        similarities = matrix @ matrix.T
        for row, chunk_id in enumerate(chunk_ids):
            candidates = [
                (float(similarities[row, other]), chunk_ids[other])
                for other in range(len(chunk_ids))
                if other != row
            ]
            candidates.sort(key=lambda value: (-value[0], value[1]))
            for _, neighbor in candidates[: self.config.semantic_neighbors]:
                left_sentence, right_sentence = self._best_sentence_pair(
                    units_by_chunk[chunk_id], units_by_chunk[neighbor], source_embeddings
                )
                edge_id = f"semantic-{_stable_digest(chunk_id, neighbor)}"
                if edge_id in edges:
                    edge_id = f"{edge_id}-{_stable_digest(chunk_id, neighbor, 'reverse')}"
                edges[edge_id] = EvidenceEdge(
                    edge_id=edge_id,
                    u=min(chunk_id, neighbor),
                    v=max(chunk_id, neighbor),
                    relation_type="semantic_bridge",
                    relation_key=chunk_id,
                    evidence=(left_sentence, right_sentence),
                    metadata={"source_chunk": chunk_id, "neighbor_chunk": neighbor},
                )

