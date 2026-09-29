"""Serializable data structures shared by indexing, retrieval, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np


@dataclass(frozen=True)
class DocumentRecord:
    """A corpus document.

    ``paragraphs`` is optional for generic corpora.  When present, it is used
    to retain paragraph-level support IDs for MuSiQue-style evaluation.
    """

    doc_id: str
    title: str
    text: str
    paragraphs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "paragraphs": list(self.paragraphs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentRecord":
        paragraphs = tuple(str(x) for x in value.get("paragraphs", ()) if str(x).strip())
        text = str(value.get("text", ""))
        if not text and paragraphs:
            text = "\n\n".join(paragraphs)
        return cls(
            doc_id=str(value["doc_id"]),
            title=str(value.get("title", value["doc_id"])),
            text=text,
            paragraphs=paragraphs,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SentenceUnit:
    """One source sentence retained in the reusable index."""

    unit_id: str
    doc_id: str
    title: str
    text: str
    sentence_id: int
    paragraph_id: int = 0

    @property
    def paragraph_unit_id(self) -> str:
        return f"{self.doc_id}:paragraph:{self.paragraph_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "sentence_id": self.sentence_id,
            "paragraph_id": self.paragraph_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SentenceUnit":
        return cls(
            unit_id=str(value["unit_id"]),
            doc_id=str(value["doc_id"]),
            title=str(value.get("title", "")),
            text=str(value["text"]),
            sentence_id=int(value.get("sentence_id", 0)),
            paragraph_id=int(value.get("paragraph_id", 0)),
        )


# The paper uses the name Evidence(e) for the source sentences carried by an
# edge.  Keeping a separate immutable type makes it impossible for query-time
# code to accidentally mutate the offline source record.
EvidenceSentence = SentenceUnit


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    sentence_ids: tuple[int, ...]
    sentence_unit_ids: tuple[str, ...]
    token_count: int
    entities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "sentence_ids": list(self.sentence_ids),
            "sentence_unit_ids": list(self.sentence_unit_ids),
            "token_count": self.token_count,
            "entities": list(self.entities),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            doc_id=str(value["doc_id"]),
            title=str(value.get("title", "")),
            text=str(value["text"]),
            sentence_ids=tuple(int(x) for x in value.get("sentence_ids", ())),
            sentence_unit_ids=tuple(str(x) for x in value.get("sentence_unit_ids", ())),
            token_count=int(value.get("token_count", 0)),
            entities=tuple(str(x) for x in value.get("entities", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class EvidenceEdge:
    edge_id: str
    u: str
    v: str
    relation_type: str
    evidence: tuple[EvidenceSentence, ...]
    embedding: np.ndarray | None = None
    relation_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def endpoints(self) -> tuple[str, str]:
        return (self.u, self.v)

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.u, self.v)))  # type: ignore[return-value]

    @property
    def evidence_text(self) -> str:
        return "\n".join(sentence.text for sentence in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "u": self.u,
            "v": self.v,
            "relation_type": self.relation_type,
            "evidence": [sentence.to_dict() for sentence in self.evidence],
            "relation_key": self.relation_key,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], embedding: np.ndarray | None = None) -> "EvidenceEdge":
        return cls(
            edge_id=str(value["edge_id"]),
            u=str(value["u"]),
            v=str(value["v"]),
            relation_type=str(value["relation_type"]),
            evidence=tuple(SentenceUnit.from_dict(x) for x in value.get("evidence", ())),
            embedding=embedding,
            relation_key=value.get("relation_key"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class EvidenceGraph:
    """Offline graph plus source-unit embeddings used by baselines."""

    chunks: dict[str, Chunk]
    edges: dict[str, EvidenceEdge]
    source_units: dict[str, SentenceUnit]
    chunk_embeddings: dict[str, np.ndarray]
    source_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> list[str]:
        return sorted(self.chunks)

    def edge_ids(self) -> list[str]:
        return sorted(self.edges)

    def adjacency(self, edge_ids: Iterable[str] | None = None) -> dict[str, list[EvidenceEdge]]:
        adjacency: dict[str, list[EvidenceEdge]] = {node: [] for node in self.chunks}
        selected = self.edges.values() if edge_ids is None else (self.edges[eid] for eid in edge_ids)
        for edge in selected:
            adjacency.setdefault(edge.u, []).append(edge)
            adjacency.setdefault(edge.v, []).append(edge)
        for node in adjacency:
            adjacency[node].sort(key=lambda edge: edge.edge_id)
        return adjacency

    def to_files(self, output_dir: str | Path) -> None:
        """Write metadata as JSON and vectors as compressed NumPy arrays."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        chunk_ids = sorted(self.chunks)
        edge_ids = sorted(self.edges)
        source_ids = sorted(self.source_units)

        def stack(mapping: Mapping[str, np.ndarray], ids: list[str]) -> np.ndarray:
            if not ids:
                return np.empty((0, 0), dtype=np.float32)
            vectors = [np.asarray(mapping[item], dtype=np.float32) for item in ids]
            return np.stack(vectors).astype(np.float32, copy=False)

        np.savez_compressed(
            output / "vectors.npz",
            chunk_embeddings=stack(self.chunk_embeddings, chunk_ids),
            edge_embeddings=stack(
                {edge_id: self.edges[edge_id].embedding for edge_id in edge_ids}, edge_ids
            ),
            source_embeddings=stack(self.source_embeddings, source_ids),
        )
        metadata = {
            "format_version": 1,
            "chunks": [self.chunks[chunk_id].to_dict() for chunk_id in chunk_ids],
            "edges": [self.edges[edge_id].to_dict() for edge_id in edge_ids],
            "source_units": [self.source_units[source_id].to_dict() for source_id in source_ids],
            "chunk_ids": chunk_ids,
            "edge_ids": edge_ids,
            "source_ids": source_ids,
            "metadata": self.metadata,
        }
        (output / "index.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save = to_files

    @classmethod
    def from_files(cls, input_dir: str | Path) -> "EvidenceGraph":
        input_path = Path(input_dir)
        metadata = json.loads((input_path / "index.json").read_text(encoding="utf-8"))
        vectors = np.load(input_path / "vectors.npz")
        chunk_ids = [str(x) for x in metadata.get("chunk_ids", ())]
        edge_ids = [str(x) for x in metadata.get("edge_ids", ())]
        source_ids = [str(x) for x in metadata.get("source_ids", ())]
        chunks = {item["chunk_id"]: Chunk.from_dict(item) for item in metadata.get("chunks", ())}
        source_units = {
            item["unit_id"]: SentenceUnit.from_dict(item)
            for item in metadata.get("source_units", ())
        }
        chunk_vectors = np.asarray(vectors["chunk_embeddings"], dtype=np.float32)
        edge_vectors = np.asarray(vectors["edge_embeddings"], dtype=np.float32)
        source_vectors = np.asarray(vectors["source_embeddings"], dtype=np.float32)
        chunk_embeddings = {
            chunk_id: chunk_vectors[index] for index, chunk_id in enumerate(chunk_ids)
        }
        source_embeddings = {
            source_id: source_vectors[index] for index, source_id in enumerate(source_ids)
        }
        edges = {}
        for index, item in enumerate(metadata.get("edges", ())):
            edge_id = str(item["edge_id"])
            embedding = edge_vectors[index] if len(edge_vectors) else None
            edges[edge_id] = EvidenceEdge.from_dict(item, embedding=embedding)
        return cls(
            chunks=chunks,
            edges=edges,
            source_units=source_units,
            chunk_embeddings=chunk_embeddings,
            source_embeddings=source_embeddings,
            metadata=dict(metadata.get("metadata", {})),
        )

    load = from_files


@dataclass(frozen=True)
class QueryEdge:
    edge_id: str
    u: str
    v: str
    raw_score: float
    weight: float

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.u, self.v)))  # type: ignore[return-value]


@dataclass
class PackedContext:
    text: str
    token_count: int
    edge_ids: tuple[str, ...] = ()
    source_unit_ids: tuple[str, ...] = ()
    paragraph_unit_ids: tuple[str, ...] = ()
    vertex_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    context: PackedContext
    query_edges: dict[str, QueryEdge]
    top_edge_ids: tuple[str, ...]
    initial_nodes: tuple[str, ...]
    terminal_nodes: tuple[str, ...]
    active_nodes: tuple[str, ...]
    candidate_nodes: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    selected_source_unit_ids: tuple[str, ...]
    activation: dict[str, float]
    effective_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionExample:
    question_id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    documents: tuple[DocumentRecord, ...]
    support_units: tuple[str, ...]
    support_level: str = "sentence"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "question": self.question,
            "answers": list(self.answers),
            "documents": [document.to_dict() for document in self.documents],
            "support_units": list(self.support_units),
            "support_level": self.support_level,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuestionExample":
        return cls(
            question_id=str(value["question_id"]),
            dataset=str(value.get("dataset", "unknown")),
            question=str(value["question"]),
            answers=tuple(str(x) for x in value.get("answers", ())),
            documents=tuple(DocumentRecord.from_dict(x) for x in value.get("documents", ())),
            support_units=tuple(str(x) for x in value.get("support_units", ())),
            support_level=str(value.get("support_level", "sentence")),
            metadata=dict(value.get("metadata", {})),
        )
