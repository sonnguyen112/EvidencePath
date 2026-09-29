"""Context assembly and hard-cap packing."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import re

from .text import TokenizerProtocol, normalize_text
from .types import EvidenceEdge, EvidenceGraph, PackedContext, QueryEdge


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    header: str
    text: str
    source_unit_ids: tuple[str, ...] = ()
    paragraph_unit_ids: tuple[str, ...] = ()
    score: float = 0.0
    vertex_ids: tuple[str, ...] = ()

    @property
    def rendered(self) -> str:
        return f"{self.header}\n{self.text}" if self.text else self.header


def _unit_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).casefold()


class ContextPacker:
    """Pack whole evidence blocks without exceeding a token budget."""

    def __init__(self, tokenizer: TokenizerProtocol, graph: EvidenceGraph | None = None) -> None:
        self.tokenizer = tokenizer
        self.graph = graph

    def edge_blocks(
        self,
        edges: Iterable[EvidenceEdge],
        scores: Mapping[str, float] | None = None,
    ) -> list[TextBlock]:
        score_map = scores or {}
        blocks: list[TextBlock] = []
        for edge in edges:
            evidence_lines: list[str] = []
            source_ids: list[str] = []
            paragraph_ids: list[str] = []
            for source in edge.evidence:
                evidence_lines.append(
                    f"[{source.title} | sentence={source.sentence_id} | paragraph={source.paragraph_id}] "
                    f"{source.text}"
                )
                source_ids.append(source.unit_id)
                paragraph_ids.append(source.paragraph_unit_id)
            blocks.append(
                TextBlock(
                    block_id=edge.edge_id,
                    header=(
                        f"[Evidence edge {edge.edge_id} | relation={edge.relation_type}]"
                    ),
                    text="\n".join(evidence_lines),
                    source_unit_ids=tuple(source_ids),
                    paragraph_unit_ids=tuple(paragraph_ids),
                    score=float(score_map.get(edge.edge_id, 0.0)),
                    vertex_ids=(edge.u, edge.v),
                )
            )
        return blocks

    def pack_edge_evidence(
        self,
        edges: Iterable[EvidenceEdge],
        *,
        scores: Mapping[str, float] | None = None,
        cap: int,
        order: str = "score",
        vertex_ids: Iterable[str] = (),
    ) -> PackedContext:
        blocks = self.edge_blocks(edges, scores=scores)
        return self.pack_blocks(blocks, cap=cap, order=order, vertex_ids=vertex_ids)

    def pack_blocks(
        self,
        blocks: Sequence[TextBlock],
        *,
        cap: int,
        order: str = "score",
        vertex_ids: Iterable[str] = (),
    ) -> PackedContext:
        if cap <= 0:
            raise ValueError("The context cap must be positive")
        ordered = self._order_blocks(blocks, order=order)
        retained: list[TextBlock] = []
        retained_lines: list[str] = []
        seen_sentence_text: set[str] = set()
        retained_sources: list[str] = []
        retained_paragraphs: list[str] = []
        current_tokens = 0

        for block in ordered:
            filtered_lines: list[str] = []
            filtered_source_ids: list[str] = []
            filtered_paragraph_ids: list[str] = []
            candidate_seen_sentence_text = set(seen_sentence_text)
            for line, source_id, paragraph_id in zip(
                block.text.splitlines(), block.source_unit_ids, block.paragraph_unit_ids
            ):
                sentence_text = line.split("] ", 1)[-1]
                key = _unit_key(sentence_text)
                if key in candidate_seen_sentence_text:
                    continue
                candidate_seen_sentence_text.add(key)
                filtered_lines.append(line)
                filtered_source_ids.append(source_id)
                filtered_paragraph_ids.append(paragraph_id)

            filtered_block = TextBlock(
                block_id=block.block_id,
                header=block.header,
                text="\n".join(filtered_lines),
                source_unit_ids=tuple(filtered_source_ids),
                paragraph_unit_ids=tuple(dict.fromkeys(filtered_paragraph_ids)),
                score=block.score,
                vertex_ids=block.vertex_ids,
            )
            if not filtered_block.text:
                continue
            rendered = filtered_block.rendered
            separator_tokens = self.tokenizer.count("\n\n") if retained else 0
            candidate_tokens = current_tokens + separator_tokens + self.tokenizer.count(rendered)
            if candidate_tokens > cap:
                continue
            retained.append(filtered_block)
            retained_lines.append(rendered)
            seen_sentence_text = candidate_seen_sentence_text
            current_tokens = candidate_tokens
            retained_sources.extend(filtered_block.source_unit_ids)
            retained_paragraphs.extend(filtered_block.paragraph_unit_ids)

        return PackedContext(
            text="\n\n".join(retained_lines),
            token_count=current_tokens,
            edge_ids=tuple(block.block_id for block in retained),
            source_unit_ids=tuple(dict.fromkeys(retained_sources)),
            paragraph_unit_ids=tuple(dict.fromkeys(retained_paragraphs)),
            vertex_ids=tuple(sorted(set(vertex_ids))),
            metadata={"order": order, "candidate_blocks": len(blocks)},
        )

    @staticmethod
    def _order_blocks(blocks: Sequence[TextBlock], *, order: str) -> list[TextBlock]:
        if order == "score":
            return sorted(blocks, key=lambda block: (-block.score, block.block_id))
        if order != "traversal":
            raise ValueError(f"Unknown block order: {order}")

        by_node: dict[str, list[TextBlock]] = defaultdict(list)
        for block in blocks:
            for node in block.vertex_ids:
                by_node[node].append(block)
        unvisited = {block.block_id: block for block in blocks}
        result: list[TextBlock] = []
        while unvisited:
            start = max(unvisited.values(), key=lambda block: (block.score, block.block_id))
            if not start.vertex_ids:
                result.append(start)
                del unvisited[start.block_id]
                continue
            start_node = min(start.vertex_ids)
            stack = [start_node]
            seen_nodes: set[str] = set()
            while stack:
                node = stack.pop()
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                incident = [
                    block
                    for block in by_node.get(node, ())
                    if block.block_id in unvisited
                ]
                incident.sort(key=lambda block: (-block.score, block.block_id))
                for block in incident:
                    if block.block_id not in unvisited:
                        continue
                    result.append(block)
                    del unvisited[block.block_id]
                    for neighbor in reversed(block.vertex_ids):
                        if neighbor not in seen_nodes:
                            stack.append(neighbor)
        return result


def generic_blocks_from_chunks(
    graph: EvidenceGraph,
    chunk_ids: Iterable[str],
    scores: Mapping[str, float],
) -> list[TextBlock]:
    """Create full-chunk blocks for the vanilla and hierarchical baselines."""

    blocks: list[TextBlock] = []
    for chunk_id in chunk_ids:
        chunk = graph.chunks[chunk_id]
        sources = [graph.source_units[item] for item in chunk.sentence_unit_ids]
        source_lines = [
            f"[{source.title} | sentence={source.sentence_id} | paragraph={source.paragraph_id}] "
            f"{source.text}"
            for source in sources
        ]
        blocks.append(
            TextBlock(
                block_id=chunk_id,
                header=f"[Retrieved chunk {chunk_id} | title={chunk.title}]",
                text="\n".join(source_lines) if source_lines else chunk.text,
                source_unit_ids=tuple(source.unit_id for source in sources),
                paragraph_unit_ids=tuple(source.paragraph_unit_id for source in sources),
                score=float(scores.get(chunk_id, 0.0)),
                vertex_ids=(chunk_id,),
            )
        )
    return blocks


def generic_blocks_from_sentences(
    graph: EvidenceGraph,
    source_ids: Iterable[str],
    scores: Mapping[str, float],
) -> list[TextBlock]:
    """Create source-sentence blocks for the independent sentence baseline."""

    blocks: list[TextBlock] = []
    for source_id in source_ids:
        source = graph.source_units[source_id]
        blocks.append(
            TextBlock(
                block_id=source_id,
                header=f"[Retrieved sentence {source_id} | title={source.title}]",
                text=source.text,
                source_unit_ids=(source.unit_id,),
                paragraph_unit_ids=(source.paragraph_unit_id,),
                score=float(scores.get(source_id, 0.0)),
            )
        )
    return blocks
