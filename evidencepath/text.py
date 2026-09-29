"""Text normalization, token counting, sentence splitting, and chunking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re
import unicodedata

from .types import Chunk, DocumentRecord, SentenceUnit


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’\)\]]+)?\s+(?=[A-Z0-9À-ÖØ-Þ])")
_ENTITY_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    """Normalize Unicode, whitespace, and surrounding text punctuation."""

    normalized = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
    return _SPACE_RE.sub(" ", normalized).strip()


def normalize_title(value: str) -> str:
    """Return the normalized title key used for document deduplication."""

    return normalize_text(value).casefold()


def normalize_entity(value: str) -> str:
    """Apply the entity matching normalization from the paper."""

    normalized = normalize_text(value).casefold()
    normalized = _ENTITY_PUNCT_RE.sub("", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def simple_tokenize(value: str) -> list[str]:
    """Tokenize text with a deterministic fallback tokenizer."""

    return _TOKEN_RE.findall(normalize_text(value))


class TokenizerProtocol:
    """Small tokenizer interface used by the core without importing Transformers."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        raise NotImplementedError

    def count(self, text: str) -> int:
        return len(self.encode(text, add_special_tokens=False))


class SimpleTokenizer(TokenizerProtocol):
    """A deterministic token counter for local tests and dependency-light runs."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(simple_tokenize(text))))


class HuggingFaceTokenizer(TokenizerProtocol):
    """Lazy wrapper around a Hugging Face tokenizer."""

    def __init__(self, model_name_or_path: str, *, max_length: int | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Hugging Face tokenization requires `transformers`; install the model extras."
            ) from exc
        self.name_or_path = model_name_or_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        kwargs = {"add_special_tokens": add_special_tokens}
        if self.max_length is not None:
            kwargs.update({"truncation": True, "max_length": self.max_length})
        return list(self.tokenizer.encode(text, **kwargs))


def split_sentences(text: str) -> list[str]:
    """Split a paragraph while keeping punctuation attached to each sentence.

    The datasets used by the paper already provide sentence boundaries for
    HotpotQA and 2WikiMultiHopQA.  This function is used for generic corpora
    and for MuSiQue paragraphs, where a conservative punctuation-based split is
    preferable to dropping source text.
    """

    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces = _SENTENCE_RE.split(normalized)
    sentences: list[str] = []
    for piece in pieces:
        piece = normalize_text(piece)
        if piece:
            sentences.append(piece)
    return sentences


def document_sentences(document: DocumentRecord) -> list[SentenceUnit]:
    """Convert a document into source sentence units with stable IDs."""

    paragraphs = document.paragraphs or tuple(
        paragraph for paragraph in re.split(r"\n\s*\n", document.text) if paragraph.strip()
    )
    if not paragraphs:
        paragraphs = (document.text,)
    units: list[SentenceUnit] = []
    sentence_id = 0
    for paragraph_id, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            unit_id = f"{document.doc_id}:sentence:{sentence_id}"
            units.append(
                SentenceUnit(
                    unit_id=unit_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=sentence,
                    sentence_id=sentence_id,
                    paragraph_id=paragraph_id,
                )
            )
            sentence_id += 1
    return units


def deduplicate_documents(documents: Iterable[DocumentRecord]) -> list[DocumentRecord]:
    """Deduplicate documents by normalized title and normalized text."""

    seen: set[tuple[str, str]] = set()
    result: list[DocumentRecord] = []
    for document in documents:
        key = (normalize_title(document.title), normalize_text(document.text).casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def _chunk_from_units(
    document: DocumentRecord,
    units: Sequence[SentenceUnit],
    chunk_index: int,
    tokenizer: TokenizerProtocol,
) -> Chunk:
    text = " ".join(unit.text for unit in units)
    return Chunk(
        chunk_id=f"{document.doc_id}:chunk:{chunk_index}",
        doc_id=document.doc_id,
        title=document.title,
        text=text,
        sentence_ids=tuple(unit.sentence_id for unit in units),
        sentence_unit_ids=tuple(unit.unit_id for unit in units),
        token_count=tokenizer.count(text),
    )


def _split_long_unit(
    document: DocumentRecord,
    unit: SentenceUnit,
    chunk_index: int,
    tokenizer: TokenizerProtocol,
    max_tokens: int,
) -> list[Chunk]:
    tokens = simple_tokenize(unit.text)
    if len(tokens) <= max_tokens:
        return [_chunk_from_units(document, [unit], chunk_index, tokenizer)]
    chunks: list[Chunk] = []
    for offset in range(0, len(tokens), max_tokens):
        text = " ".join(tokens[offset : offset + max_tokens])
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:chunk:{chunk_index + len(chunks)}",
                doc_id=document.doc_id,
                title=document.title,
                text=text,
                sentence_ids=(unit.sentence_id,),
                sentence_unit_ids=(unit.unit_id,),
                token_count=tokenizer.count(text),
                metadata={"long_sentence_split": True, "token_offset": offset},
            )
        )
    return chunks


def sentence_aware_chunks(
    document: DocumentRecord,
    tokenizer: TokenizerProtocol,
    *,
    max_tokens: int = 256,
    overlap_tokens: int = 64,
) -> tuple[list[Chunk], list[SentenceUnit]]:
    """Create approximately fixed-size chunks while preferring sentence boundaries.

    Overlap is measured in fallback/Hugging Face tokens and is formed from the
    trailing sentences of the previous chunk.  A sentence longer than the
    budget is split into token windows so the chunking procedure always makes
    progress.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens")

    units = document_sentences(document)
    if not units:
        return [], []

    chunks: list[Chunk] = []
    current: list[SentenceUnit] = []
    current_tokens = 0
    chunk_index = 0
    unit_index = 0

    while unit_index < len(units):
        unit = units[unit_index]
        unit_tokens = tokenizer.count(unit.text)
        if unit_tokens > max_tokens and not current:
            long_chunks = _split_long_unit(
                document, unit, chunk_index, tokenizer, max_tokens=max_tokens
            )
            chunks.extend(long_chunks)
            chunk_index += len(long_chunks)
            unit_index += 1
            continue

        if current and unit_tokens > max_tokens:
            # Do not overlap into a chunk immediately before a long sentence;
            # otherwise the same trailing sentence could be reintroduced forever.
            chunks.append(_chunk_from_units(document, current, chunk_index, tokenizer))
            chunk_index += 1
            current = []
            current_tokens = 0
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(_chunk_from_units(document, current, chunk_index, tokenizer))
            chunk_index += 1
            overlap: list[SentenceUnit] = []
            overlap_count = 0
            for previous in reversed(current):
                previous_tokens = tokenizer.count(previous.text)
                if overlap_count + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_count += previous_tokens
            current = overlap
            current_tokens = overlap_count
            continue

        current.append(unit)
        current_tokens += unit_tokens
        unit_index += 1

    if current:
        chunks.append(_chunk_from_units(document, current, chunk_index, tokenizer))
    return chunks, units
