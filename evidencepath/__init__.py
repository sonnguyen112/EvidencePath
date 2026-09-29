"""EvidencePath-RAG implementation.

The package intentionally keeps the graph and evaluation layers independent of
the model backend.  Model-heavy components are imported lazily so that graph
serialization, metrics, and deterministic unit tests work without GPU packages.
"""

from .types import (
    Chunk,
    DocumentRecord,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceSentence,
    PackedContext,
    QuestionExample,
    RetrievalResult,
    SentenceUnit,
)

__all__ = [
    "Chunk",
    "DocumentRecord",
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceSentence",
    "PackedContext",
    "QuestionExample",
    "RetrievalResult",
    "SentenceUnit",
]

__version__ = "0.1.0"
