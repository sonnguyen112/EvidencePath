"""Dataset adapters and reproducible question/corpus serialization."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from collections.abc import Iterable, Mapping, Sequence

from .text import normalize_text, normalize_title
from .types import DocumentRecord, QuestionExample


def _digest(*values: str) -> str:
    return hashlib.sha1("\x1f".join(values).encode("utf-8")).hexdigest()[:20]


def document_id(title: str, text: str) -> str:
    """Generate a stable document ID shared by question and corpus files."""

    return f"doc-{_digest(normalize_title(title), normalize_text(text).casefold())}"


def read_records(path: str | Path) -> list[dict]:
    """Read a JSON list, JSON object with a records field, or JSONL file."""

    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.casefold() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    value = json.loads(raw)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, dict):
        for key in ("data", "records", "examples", "questions"):
            if isinstance(value.get(key), list):
                return [dict(item) for item in value[key]]
        return [value]
    raise ValueError(f"Unsupported JSON structure in {source}")


def write_jsonl(path: str | Path, records: Iterable[Mapping]) -> None:
    """Write one JSON object per line."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _title_sentences(item: object) -> tuple[str, tuple[str, ...]]:
    if isinstance(item, Mapping):
        title = item.get("title", item.get("name", ""))
        sentences = item.get("sentences", item.get("text", item.get("paragraph_text", "")))
        if isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)):
            return str(title), tuple(str(value) for value in sentences)
        return str(title), (str(sentences),)
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
        title = str(item[0])
        sentences = item[1]
        if isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)):
            return title, tuple(str(value) for value in sentences)
        return title, (str(sentences),)
    return "", (str(item),)


def _context_documents(raw: Mapping) -> list[DocumentRecord]:
    context = raw.get("context", raw.get("documents", raw.get("paragraphs", [])))
    documents: list[DocumentRecord] = []
    if isinstance(context, Mapping):
        context = list(context.items())
    if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
        context = [context]
    for item in context:
        title, values = _title_sentences(item)
        title = title or "untitled"
        values = tuple(value for value in values if normalize_text(value))
        if not values:
            continue
        text = " ".join(values)
        documents.append(
            DocumentRecord(
                doc_id=document_id(title, text),
                title=title,
                text=text,
                paragraphs=values,
            )
        )
    return documents


def _answer_values(raw: Mapping) -> tuple[str, ...]:
    value = raw.get("answers", raw.get("answer", raw.get("gold", "")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(str(item) for item in value if str(item).strip())
    else:
        values = (str(value),) if str(value).strip() else ()
    return values


def _support_from_pairs(raw: Mapping, documents: Sequence[DocumentRecord]) -> tuple[str, ...]:
    by_title = {normalize_title(document.title): document for document in documents}
    support = raw.get("supporting_facts", raw.get("support", raw.get("evidence", [])))
    if isinstance(support, Mapping):
        support = support.get("facts", support.get("sentences", support.get("paragraphs", [])))
    if not isinstance(support, Sequence) or isinstance(support, (str, bytes)):
        support = [support]
    result: list[str] = []
    for item in support:
        title = ""
        index = None
        if isinstance(item, Mapping):
            title = str(item.get("title", item.get("doc_title", item.get("name", ""))))
            index = item.get("sent_id", item.get("sentence_id", item.get("idx", item.get("index"))))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            title, index = str(item[0]), item[1]
        if title == "" and isinstance(item, str):
            title = item
        document = by_title.get(normalize_title(title))
        if document is None:
            continue
        if index is None:
            result.append(f"{document.doc_id}:paragraph:0")
        else:
            try:
                result.append(f"{document.doc_id}:sentence:{int(index)}")
            except (TypeError, ValueError):
                result.append(f"{document.doc_id}:paragraph:0")
    return tuple(dict.fromkeys(result))


def _adapt_hotpot(raw: Mapping, dataset: str) -> QuestionExample:
    documents = _context_documents(raw)
    support = _support_from_pairs(raw, documents)
    question_id = str(raw.get("_id", raw.get("id", raw.get("question_id", ""))))
    if not question_id:
        question_id = f"q-{_digest(str(raw.get('question', '')), str(raw.get('answer', '')))}"
    return QuestionExample(
        question_id=question_id,
        dataset=dataset,
        question=str(raw.get("question", "")),
        answers=_answer_values(raw),
        documents=tuple(documents),
        support_units=support,
        support_level="sentence",
        metadata={"source_format": "hotpotqa"},
    )


def _adapt_musique(raw: Mapping, dataset: str) -> QuestionExample:
    paragraphs = raw.get("paragraphs", raw.get("context", []))
    documents: list[DocumentRecord] = []
    support: list[str] = []
    if isinstance(paragraphs, Sequence) and not isinstance(paragraphs, (str, bytes)):
        for index, item in enumerate(paragraphs):
            if isinstance(item, Mapping):
                title = str(item.get("title", item.get("name", f"paragraph-{index}")))
                text = str(item.get("paragraph_text", item.get("text", item.get("paragraph", ""))))
                is_supporting = bool(
                    item.get("is_supporting", item.get("supporting", item.get("is_support", False)))
                )
            else:
                title, values = _title_sentences(item)
                text = " ".join(values)
                is_supporting = False
            if not normalize_text(text):
                continue
            document = DocumentRecord(
                doc_id=document_id(title, text),
                title=title,
                text=text,
                paragraphs=(text,),
                metadata={"musique_paragraph_index": index},
            )
            documents.append(document)
            if is_supporting:
                support.append(f"{document.doc_id}:paragraph:0")
    explicit_support = raw.get("supporting_paragraphs", raw.get("supporting_facts"))
    if explicit_support and not support:
        if isinstance(explicit_support, Sequence) and not isinstance(explicit_support, (str, bytes)):
            for value in explicit_support:
                if isinstance(value, int) and 0 <= value < len(documents):
                    support.append(f"{documents[value].doc_id}:paragraph:0")
                elif isinstance(value, Mapping):
                    title = str(value.get("title", ""))
                    for document in documents:
                        if normalize_title(document.title) == normalize_title(title):
                            support.append(f"{document.doc_id}:paragraph:0")
                            break
                elif isinstance(value, str):
                    for document in documents:
                        if normalize_title(document.title) == normalize_title(value):
                            support.append(f"{document.doc_id}:paragraph:0")
                            break
    question_id = str(raw.get("id", raw.get("question_id", "")))
    if not question_id:
        question_id = f"q-{_digest(str(raw.get('question', '')), str(raw.get('answer', '')))}"
    return QuestionExample(
        question_id=question_id,
        dataset=dataset,
        question=str(raw.get("question", "")),
        answers=_answer_values(raw),
        documents=tuple(documents),
        support_units=tuple(dict.fromkeys(support)),
        support_level="paragraph",
        metadata={"source_format": "musique"},
    )


def _adapt_generic(raw: Mapping, dataset: str) -> QuestionExample:
    documents = _context_documents(raw)
    support = _support_from_pairs(raw, documents)
    explicit_documents = raw.get("documents")
    if not documents and isinstance(explicit_documents, Sequence):
        documents = [DocumentRecord.from_dict(value) for value in explicit_documents]
    question_id = str(raw.get("id", raw.get("question_id", "")))
    if not question_id:
        question_id = f"q-{_digest(str(raw.get('question', '')), str(raw.get('answer', '')))}"
    return QuestionExample(
        question_id=question_id,
        dataset=dataset,
        question=str(raw.get("question", "")),
        answers=_answer_values(raw),
        documents=tuple(documents),
        support_units=support,
        support_level=str(raw.get("support_level", "sentence")),
        metadata={"source_format": "generic"},
    )


def adapt_record(raw: Mapping, dataset: str) -> QuestionExample:
    """Normalize one supported benchmark record."""

    name = dataset.casefold().replace("_", "-")
    if "musique" in name:
        return _adapt_musique(raw, dataset)
    if "hotpot" in name or "2wiki" in name or "2-wiki" in name:
        return _adapt_hotpot(raw, dataset)
    return _adapt_generic(raw, dataset)


def load_examples(
    *,
    dataset: str,
    input_path: str | Path | None = None,
    hf_dataset: str | None = None,
    split: str = "validation",
    hf_config: str | None = None,
) -> list[QuestionExample]:
    """Load and adapt local records or a Hugging Face dataset split."""

    if input_path is not None:
        raw_records = read_records(input_path)
    else:
        if not hf_dataset:
            raise ValueError("Provide either input_path or hf_dataset")
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Hugging Face dataset loading requires `datasets`.") from exc
        loaded = load_dataset(hf_dataset, hf_config, split=split)
        raw_records = [dict(record) for record in loaded]
    return [adapt_record(record, dataset) for record in raw_records]


def sample_examples(
    examples: Sequence[QuestionExample], *, sample_size: int | None, seed: int
) -> list[QuestionExample]:
    """Sample a fixed evaluation set without changing question IDs."""

    if sample_size is None or sample_size >= len(examples):
        return list(examples)
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples)), sample_size))
    return [examples[index] for index in indices]


def corpus_from_examples(examples: Iterable[QuestionExample]) -> list[DocumentRecord]:
    """Build a deduplicated corpus from question-local contexts."""

    by_key: dict[tuple[str, str], DocumentRecord] = {}
    for example in examples:
        for document in example.documents:
            key = (normalize_title(document.title), normalize_text(document.text).casefold())
            by_key.setdefault(key, document)
    return [by_key[key] for key in sorted(by_key)]


def write_prepared_examples(
    output_dir: str | Path,
    examples: Sequence[QuestionExample],
    *,
    corpus: Sequence[DocumentRecord] | None = None,
) -> None:
    """Write questions, corpus, and a manifest for later identical evaluation sets."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "questions.jsonl", (example.to_dict() for example in examples))
    corpus_records = list(corpus) if corpus is not None else corpus_from_examples(examples)
    write_jsonl(output / "corpus.jsonl", (document.to_dict() for document in corpus_records))
    manifest = {
        "num_questions": len(examples),
        "question_ids": [example.question_id for example in examples],
        "num_documents": len(corpus_records),
        "dataset": examples[0].dataset if examples else "unknown",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_prepared_questions(path: str | Path) -> list[QuestionExample]:
    """Read the question JSONL produced by ``prepare_datasets.py``."""

    return [QuestionExample.from_dict(record) for record in read_records(path)]


def load_corpus(path: str | Path) -> list[DocumentRecord]:
    """Read corpus JSONL or JSON produced by the preparation script."""

    return [DocumentRecord.from_dict(record) for record in read_records(path)]
