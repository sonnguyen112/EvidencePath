"""Entity extraction backends for offline evidence-graph construction."""

from __future__ import annotations

from collections.abc import Iterable
import re

from .text import normalize_entity


class EntityExtractor:
    """Entity extraction protocol."""

    def extract(self, text: str) -> list[str]:
        raise NotImplementedError


class RegexEntityExtractor(EntityExtractor):
    """Conservative fallback extractor for environments without a NER model."""

    _proper_name = re.compile(
        r"\b(?:[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){0,5})\b", flags=re.UNICODE
    )
    _acronym = re.compile(r"\b[A-Z]{2,}(?:[-\s][A-Z0-9]{2,})*\b")

    def extract(self, text: str) -> list[str]:
        values = set()
        for match in self._proper_name.findall(text):
            normalized = normalize_entity(match)
            if len(normalized) >= 2:
                values.add(normalized)
        for match in self._acronym.findall(text):
            normalized = normalize_entity(match)
            if len(normalized) >= 2:
                values.add(normalized)
        return sorted(values)


class SpacyEntityExtractor(EntityExtractor):
    """spaCy NER extractor with deterministic normalized output."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The spaCy entity backend requires `spacy`.") from exc
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name, exclude=["parser", "lemmatizer", "textcat"])
        except OSError as exc:  # pragma: no cover - environment dependent
            raise OSError(
                f"spaCy model {model_name!r} is not installed; install it or use regex extraction."
            ) from exc

    def extract(self, text: str) -> list[str]:
        doc = self.nlp(text)
        values = {normalize_entity(entity.text) for entity in doc.ents}
        return sorted(value for value in values if len(value) >= 2)


class CompositeEntityExtractor(EntityExtractor):
    """Run several extractors and return their union."""

    def __init__(self, extractors: Iterable[EntityExtractor]) -> None:
        self.extractors = tuple(extractors)

    def extract(self, text: str) -> list[str]:
        values: set[str] = set()
        for extractor in self.extractors:
            values.update(extractor.extract(text))
        return sorted(values)


def build_entity_extractor(backend: str, *, spacy_model: str = "en_core_web_sm") -> EntityExtractor:
    """Create an entity extractor from a CLI/configuration name."""

    backend = backend.casefold()
    if backend == "regex":
        return RegexEntityExtractor()
    if backend == "spacy":
        return SpacyEntityExtractor(spacy_model)
    if backend in {"spacy+regex", "regex+spacy"}:
        return CompositeEntityExtractor(
            [SpacyEntityExtractor(spacy_model), RegexEntityExtractor()]
        )
    raise ValueError(f"Unknown entity backend: {backend}")

