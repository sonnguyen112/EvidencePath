"""Embedding backends used by offline indexing and query-time scoring."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import math
import re

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows, leaving zero rows unchanged."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, np.maximum(norms, 1e-12), out=np.zeros_like(array), where=norms > 0)


class EmbeddingEncoder:
    """Minimal encoder protocol."""

    dimension: int

    def encode(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEncoder(EmbeddingEncoder):
    """Deterministic lexical encoder for tests and dependency-light development."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def encode(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            row = np.zeros(self.dimension, dtype=np.float32)
            tokens = self._token_pattern.findall(str(text).casefold())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                index = value % self.dimension
                sign = 1.0 if (value >> 17) & 1 else -1.0
                row[index] += sign
                second = (value >> 29) % self.dimension
                row[second] += 0.5 * sign
            rows.append(row)
        if not rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack(rows))


class SentenceTransformerEncoder(EmbeddingEncoder):
    """Sentence-transformers backend with the paper's normalized batch behavior."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int = 512,
        half_precision: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The sentence-transformers backend requires `sentence-transformers`."
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model.max_seq_length = max_seq_length
        actual_device = str(getattr(self.model, "device", device or ""))
        if half_precision and actual_device.startswith("cuda"):
            self.model = self.model.half()
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            values,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class HFMeanPoolingEncoder(EmbeddingEncoder):
    """Pure Transformers mean-pooling backend for models without sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int = 512,
        half_precision: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The Hugging Face embedding backend requires `torch` and `transformers`."
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        if half_precision and self.device.startswith("cuda"):
            self.model.half()
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        rows: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                rows.append(pooled.float().cpu().numpy())
        return l2_normalize(np.concatenate(rows, axis=0))


def build_encoder(
    backend: str,
    model_name_or_path: str,
    *,
    batch_size: int = 32,
    device: str | None = None,
    max_seq_length: int = 512,
    hashing_dimension: int = 384,
) -> EmbeddingEncoder:
    """Create an encoder without importing optional model libraries eagerly."""

    backend = backend.casefold()
    if backend == "hashing":
        return HashingEncoder(dimension=hashing_dimension)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEncoder(
            model_name_or_path,
            batch_size=batch_size,
            device=device,
            max_seq_length=max_seq_length,
        )
    if backend in {"huggingface", "hf", "mean-pooling"}:
        return HFMeanPoolingEncoder(
            model_name_or_path,
            batch_size=batch_size,
            device=device,
            max_seq_length=max_seq_length,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")
