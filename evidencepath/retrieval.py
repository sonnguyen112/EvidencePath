"""Query-time EvidencePath-RAG routing and KMB Steiner selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from .embeddings import EmbeddingEncoder, l2_normalize
from .packing import ContextPacker
from .text import TokenizerProtocol
from .types import (
    EvidenceEdge,
    EvidenceGraph,
    PackedContext,
    QueryEdge,
    RetrievalResult,
)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 20
    alpha: float = 0.85
    iterations: int = 2
    threshold: float = 0.35
    terminal_multiplier: float = 0.30
    edge_cutoff: float = 0.05
    augmentation_cutoff: float = 0.60
    hard_cap_thresholds: tuple[float, ...] = (0.35, 0.30, 0.25, 0.20, 0.15)
    hard_cap_target_fraction: float = 0.90


@dataclass
class _CandidateGraph:
    nodes: set[str]
    edges: dict[str, QueryEdge]
    terminals: set[str]


class EvidencePathRetriever:
    """Implement Algorithm 1 from the paper."""

    def __init__(
        self,
        graph: EvidenceGraph,
        encoder: EmbeddingEncoder,
        tokenizer: TokenizerProtocol,
        config: RetrievalConfig | None = None,
        *,
        edge_filter: Callable[[EvidenceEdge], bool] | None = None,
    ) -> None:
        self.graph = graph
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config or RetrievalConfig()
        self.edge_filter = edge_filter or (lambda edge: True)
        self.packer = ContextPacker(tokenizer, graph)
        self._raw_edge_scores: dict[str, float] = {}

    def retrieve(
        self,
        query: str,
        *,
        context_cap: int,
        hard_cap: bool = True,
        solver: str = "kmb",
        disable_propagation: bool = False,
        pack_extra_candidates: bool = False,
    ) -> RetrievalResult:
        """Retrieve and pack evidence for one query.

        With ``hard_cap=True`` the candidate graph and Steiner selection are
        rebuilt at each threshold in the paper's adaptive schedule.  The
        score-based packer is used for these trials; the original one-threshold
        policy uses deterministic traversal order.
        """

        query_vector = l2_normalize(self.encoder.encode([query]))[0]
        query_edges = self._score_and_collapse(query_vector)
        if not query_edges:
            return self._empty_result(context_cap, threshold=self.config.threshold)

        top_edges = self._top_edges(query_edges)
        initial_nodes = {node for edge in top_edges for node in (edge.u, edge.v)}
        activation = self._initialize_activation(top_edges)
        if not disable_propagation:
            activation = self._propagate(query_edges.values(), activation)

        thresholds = (
            tuple(self.config.hard_cap_thresholds)
            if hard_cap
            else (self.config.threshold,)
        )
        last_result: RetrievalResult | None = None
        for threshold in thresholds:
            terminals = self._determine_terminals(top_edges)
            active_nodes = {
                node for node, value in activation.items() if value > float(threshold)
            }
            candidate = self._build_candidate_graph(
                query_edges,
                active_nodes=active_nodes,
                terminals=terminals,
                initial_nodes=initial_nodes,
            )
            selected_ids, selected_vertices = self._select_edges(
                candidate, solver=solver
            )
            if pack_extra_candidates and solver == "kmb":
                extras = sorted(
                    (
                        edge
                        for edge_id, edge in candidate.edges.items()
                        if edge_id not in selected_ids
                    ),
                    key=lambda edge: (-edge.weight, -edge.raw_score, edge.edge_id),
                )
                selected_ids = selected_ids + [edge.edge_id for edge in extras]
                selected_vertices.update(
                    node for edge in extras for node in (edge.u, edge.v)
                )
            selected_edges = [self.graph.edges[edge_id] for edge_id in selected_ids]
            score_map = {
                edge_id: query_edges[edge_id].weight
                for edge_id in selected_ids
                if edge_id in query_edges
            }
            order = "score" if hard_cap else "traversal"
            context = self.packer.pack_edge_evidence(
                selected_edges,
                scores=score_map,
                cap=context_cap,
                order=order,
                vertex_ids=selected_vertices,
            )
            pre_pack_source_ids = tuple(
                dict.fromkeys(
                    source.unit_id for edge in selected_edges for source in edge.evidence
                )
            )
            last_result = RetrievalResult(
                context=context,
                query_edges=query_edges,
                top_edge_ids=tuple(edge.edge_id for edge in top_edges),
                initial_nodes=tuple(sorted(initial_nodes)),
                terminal_nodes=tuple(sorted(candidate.terminals)),
                active_nodes=tuple(sorted(active_nodes)),
                candidate_nodes=tuple(sorted(candidate.nodes)),
                selected_edge_ids=tuple(selected_ids),
                selected_source_unit_ids=pre_pack_source_ids,
                activation=dict(sorted(activation.items())),
                effective_threshold=float(threshold),
                metadata={
                    "solver": solver,
                    "hard_cap": hard_cap,
                    "disable_propagation": disable_propagation,
                    "pack_extra_candidates": pack_extra_candidates,
                    "candidate_edges": len(candidate.edges),
                    "selected_vertices": sorted(selected_vertices),
                    "packed_edge_ids": list(context.edge_ids),
                },
            )
            if not hard_cap or context.token_count >= self.config.hard_cap_target_fraction * context_cap:
                return last_result
        if last_result is None:
            return self._empty_result(context_cap, threshold=self.config.threshold)
        return last_result

    def _score_and_collapse(self, query_vector: np.ndarray) -> dict[str, QueryEdge]:
        by_pair: dict[tuple[str, str], tuple[float, EvidenceEdge]] = {}
        self._raw_edge_scores = {}
        for edge_id in sorted(self.graph.edges):
            edge = self.graph.edges[edge_id]
            if not self.edge_filter(edge) or edge.embedding is None:
                continue
            raw_score = float(np.dot(query_vector, edge.embedding))
            self._raw_edge_scores[edge_id] = raw_score
            current = by_pair.get(edge.pair_key)
            if current is None or raw_score > current[0] or (
                raw_score == current[0] and edge.edge_id < current[1].edge_id
            ):
                by_pair[edge.pair_key] = (raw_score, edge)
        result: dict[str, QueryEdge] = {}
        for raw_score, edge in by_pair.values():
            result[edge.edge_id] = QueryEdge(
                edge_id=edge.edge_id,
                u=edge.u,
                v=edge.v,
                raw_score=raw_score,
                weight=max(0.0, raw_score),
            )
        return result

    def _top_edges(self, query_edges: Mapping[str, QueryEdge]) -> list[QueryEdge]:
        if self.config.top_k <= 0:
            return []
        return sorted(
            query_edges.values(),
            key=lambda edge: (-edge.weight, -edge.raw_score, edge.edge_id),
        )[: self.config.top_k]

    @staticmethod
    def _initialize_activation(top_edges: Iterable[QueryEdge]) -> dict[str, float]:
        activation: dict[str, float] = defaultdict(float)
        for edge in top_edges:
            activation[edge.u] = max(activation[edge.u], edge.weight)
            activation[edge.v] = max(activation[edge.v], edge.weight)
        return dict(activation)

    def _propagate(
        self,
        query_edges: Iterable[QueryEdge],
        initial: Mapping[str, float],
    ) -> dict[str, float]:
        activation: dict[str, float] = defaultdict(float, initial)
        edges = tuple(query_edges)
        all_nodes = set(activation)
        for edge in edges:
            all_nodes.update((edge.u, edge.v))
        for _ in range(max(0, self.config.iterations)):
            updated = defaultdict(float, activation)
            for edge in edges:
                updated[edge.u] = max(updated[edge.u], self.config.alpha * edge.weight * activation[edge.v])
                updated[edge.v] = max(updated[edge.v], self.config.alpha * edge.weight * activation[edge.u])
            activation = updated
        return {node: float(activation[node]) for node in sorted(all_nodes)}

    def _determine_terminals(self, top_edges: list[QueryEdge]) -> set[str]:
        if not top_edges:
            return set()
        maximum = max(edge.weight for edge in top_edges)
        cutoff = self.config.terminal_multiplier * maximum
        terminals = {
            node
            for edge in top_edges
            if edge.weight > cutoff
            for node in (edge.u, edge.v)
        }
        if len(terminals) >= 2:
            return terminals
        for edge in top_edges:
            terminals.update((edge.u, edge.v))
            if len(terminals) >= 2:
                break
        return terminals

    def _build_candidate_graph(
        self,
        query_edges: dict[str, QueryEdge],
        *,
        active_nodes: set[str],
        terminals: set[str],
        initial_nodes: set[str],
    ) -> _CandidateGraph:
        candidate_nodes = set(terminals) | set(active_nodes)
        candidate_edges: dict[str, QueryEdge] = {}
        for edge_id, edge in query_edges.items():
            if edge.weight <= self.config.edge_cutoff:
                continue
            if edge.u in active_nodes or edge.v in active_nodes or (
                edge.u in terminals and edge.v in terminals
            ):
                candidate_edges[edge_id] = edge
                candidate_nodes.update((edge.u, edge.v))

        terminals = set(terminals)
        if len(terminals) == 0:
            top = self._top_edges(query_edges)
            if top:
                terminals.update((top[0].u, top[0].v))
                candidate_nodes.update(terminals)
        elif len(terminals) == 1:
            only_terminal = next(iter(terminals))
            neighbors = [
                edge
                for edge in query_edges.values()
                if only_terminal in (edge.u, edge.v)
                and (edge.u if edge.v == only_terminal else edge.v) in active_nodes
                and edge.weight > self.config.edge_cutoff
            ]
            neighbors.sort(
                key=lambda edge: (
                    -edge.weight,
                    edge.u if edge.v == only_terminal else edge.v,
                    edge.edge_id,
                )
            )
            if neighbors:
                edge = neighbors[0]
                terminals.add(edge.u if edge.v == only_terminal else edge.v)
                candidate_edges[edge.edge_id] = edge
                candidate_nodes.update((edge.u, edge.v))

        active_seeds = set(initial_nodes) & set(active_nodes)
        components = self._components(candidate_nodes, candidate_edges.values())
        used_augmentation: set[str] = set()
        for component in components:
            component_terminals = terminals & component
            if not component_terminals or component & active_seeds:
                continue
            augmentable: list[QueryEdge] = []
            for edge_id, raw_score in sorted(self._raw_edge_scores.items()):
                if raw_score <= self.config.augmentation_cutoff:
                    continue
                if edge_id in candidate_edges or edge_id in used_augmentation:
                    continue
                stored_edge = self.graph.edges[edge_id]
                augmentable.append(
                    query_edges.get(
                        edge_id,
                        QueryEdge(
                            edge_id=edge_id,
                            u=stored_edge.u,
                            v=stored_edge.v,
                            raw_score=raw_score,
                            weight=max(0.0, raw_score),
                        ),
                    )
                )
            augmentable.sort(key=lambda edge: (-edge.raw_score, edge.edge_id))
            if augmentable:
                edge = augmentable[0]
                used_augmentation.add(edge.edge_id)
                query_edges.setdefault(edge.edge_id, edge)
                candidate_edges[edge.edge_id] = edge
                candidate_nodes.update((edge.u, edge.v))

        return _CandidateGraph(
            nodes=candidate_nodes,
            edges=candidate_edges,
            terminals=terminals,
        )

    @staticmethod
    def _components(nodes: Iterable[str], edges: Iterable[QueryEdge]) -> list[set[str]]:
        parent = {node: node for node in nodes}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for edge in edges:
            parent.setdefault(edge.u, edge.u)
            parent.setdefault(edge.v, edge.v)
            union(edge.u, edge.v)
        groups: dict[str, set[str]] = defaultdict(set)
        for node in sorted(parent):
            groups[find(node)].add(node)
        return [groups[root] for root in sorted(groups)]

    def _select_edges(
        self,
        candidate: _CandidateGraph,
        *,
        solver: str,
    ) -> tuple[list[str], set[str]]:
        if solver not in {"kmb", "greedy"}:
            raise ValueError(f"Unknown solver: {solver}")
        if solver == "greedy":
            ordered = sorted(
                candidate.edges.values(),
                key=lambda edge: (-edge.weight, -edge.raw_score, edge.edge_id),
            )
            selected = [edge.edge_id for edge in ordered]
            vertices = {node for edge in ordered for node in (edge.u, edge.v)}
            return selected, vertices

        selected: list[str] = []
        selected_vertices: set[str] = set()
        for component in self._components(candidate.nodes, candidate.edges.values()):
            terminals = sorted(candidate.terminals & component)
            if not terminals:
                continue
            component_edges = {
                edge.edge_id: edge
                for edge in candidate.edges.values()
                if edge.u in component and edge.v in component
            }
            if len(terminals) < 2:
                fallback = self._best_incident_edges(terminals, component_edges)
            else:
                fallback = self._kmb_steiner_tree(component, terminals, component_edges)
            selected.extend(fallback)
            for edge_id in fallback:
                edge = component_edges[edge_id]
                selected_vertices.update((edge.u, edge.v))
        selected = list(dict.fromkeys(selected))
        return selected, selected_vertices

    @staticmethod
    def _best_incident_edges(
        terminals: Iterable[str], edges: Mapping[str, QueryEdge]
    ) -> list[str]:
        selected: list[str] = []
        for terminal in sorted(terminals):
            incident = [
                edge for edge in edges.values() if terminal in (edge.u, edge.v)
            ]
            incident.sort(key=lambda edge: (-edge.weight, edge.edge_id))
            if incident:
                selected.append(incident[0].edge_id)
        return list(dict.fromkeys(selected))

    @staticmethod
    def _dijkstra(
        source: str,
        nodes: set[str],
        edges: Mapping[str, QueryEdge],
    ) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
        adjacency: dict[str, list[QueryEdge]] = defaultdict(list)
        for edge in edges.values():
            adjacency[edge.u].append(edge)
            adjacency[edge.v].append(edge)
        for values in adjacency.values():
            values.sort(key=lambda edge: edge.edge_id)
        distances = {node: float("inf") for node in nodes}
        previous: dict[str, tuple[str, str]] = {}
        distances[source] = 0.0
        heap: list[tuple[float, str]] = [(0.0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances[node] + 1e-12:
                continue
            for edge in adjacency.get(node, ()):
                neighbor = edge.v if edge.u == node else edge.u
                cost = max(0.0, 1.0 - edge.weight)
                candidate = distance + cost
                old = distances[neighbor]
                previous_key = previous.get(neighbor, ("~", "~"))
                candidate_key = (node, edge.edge_id)
                if candidate < old - 1e-12 or (
                    abs(candidate - old) <= 1e-12 and candidate_key < previous_key
                ):
                    distances[neighbor] = candidate
                    previous[neighbor] = (node, edge.edge_id)
                    heapq.heappush(heap, (candidate, neighbor))
        return distances, previous

    @classmethod
    def _path_edges(
        cls,
        source: str,
        target: str,
        previous: Mapping[str, tuple[str, str]],
    ) -> list[str] | None:
        if source == target:
            return []
        current = target
        path: list[str] = []
        seen: set[str] = set()
        while current != source:
            if current in seen or current not in previous:
                return None
            seen.add(current)
            parent, edge_id = previous[current]
            path.append(edge_id)
            current = parent
        path.reverse()
        return path

    @classmethod
    def _kmb_steiner_tree(
        cls,
        component: set[str],
        terminals: list[str],
        edges: Mapping[str, QueryEdge],
    ) -> list[str]:
        shortest: dict[str, tuple[dict[str, float], dict[str, tuple[str, str]]]] = {
            terminal: cls._dijkstra(terminal, component, edges) for terminal in terminals
        }
        metric_edges: list[tuple[float, str, str]] = []
        for index, left in enumerate(terminals):
            for right in terminals[index + 1 :]:
                distance = shortest[left][0].get(right, float("inf"))
                if distance < float("inf"):
                    metric_edges.append((distance, left, right))
        metric_edges.sort(key=lambda item: (item[0], item[1], item[2]))

        parent = {terminal: terminal for terminal in terminals}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> bool:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return False
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root
            return True

        expanded_ids: list[str] = []
        for _, left, right in metric_edges:
            if not union(left, right):
                continue
            path = cls._path_edges(left, right, shortest[left][1])
            if path is None:
                continue
            expanded_ids.extend(path)
        expanded_ids = list(dict.fromkeys(expanded_ids))
        if not expanded_ids:
            return []

        # A minimum spanning tree over the expanded union removes cycles from
        # the union of terminal shortest paths before non-terminal pruning.
        expanded_edges = {edge_id: edges[edge_id] for edge_id in expanded_ids}
        expanded_nodes = {node for edge in expanded_edges.values() for node in (edge.u, edge.v)}
        selected_ids: list[str] = []
        node_parent = {node: node for node in expanded_nodes}

        def node_find(node: str) -> str:
            while node_parent[node] != node:
                node_parent[node] = node_parent[node_parent[node]]
                node = node_parent[node]
            return node

        def node_union(left: str, right: str) -> bool:
            left_root, right_root = node_find(left), node_find(right)
            if left_root == right_root:
                return False
            if left_root < right_root:
                node_parent[right_root] = left_root
            else:
                node_parent[left_root] = right_root
            return True

        for edge in sorted(
            expanded_edges.values(), key=lambda edge: (1.0 - edge.weight, edge.edge_id)
        ):
            if node_union(edge.u, edge.v):
                selected_ids.append(edge.edge_id)

        # Remove leaves that are not required terminals.
        selected = {edge_id: expanded_edges[edge_id] for edge_id in selected_ids}
        changed = True
        while changed:
            changed = False
            degree: dict[str, int] = defaultdict(int)
            for edge in selected.values():
                degree[edge.u] += 1
                degree[edge.v] += 1
            removable = [
                node
                for node, value in degree.items()
                if value <= 1 and node not in terminals
            ]
            if removable:
                changed = True
                for node in removable:
                    for edge_id, edge in list(selected.items()):
                        if node in (edge.u, edge.v):
                            del selected[edge_id]
        return sorted(selected)

    def _empty_result(self, context_cap: int, *, threshold: float) -> RetrievalResult:
        return RetrievalResult(
            context=PackedContext(text="", token_count=0),
            query_edges={},
            top_edge_ids=(),
            initial_nodes=(),
            terminal_nodes=(),
            active_nodes=(),
            candidate_nodes=(),
            selected_edge_ids=(),
            selected_source_unit_ids=(),
            activation={},
            effective_threshold=threshold,
            metadata={"empty_graph": True, "context_cap": context_cap},
        )
