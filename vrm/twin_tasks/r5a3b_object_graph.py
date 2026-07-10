from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .r5a3b_parse_hypotheses import ParseState

EDGE_PREDICATES = (
    "same_color",
    "same_shape",
    "same_size",
    "nearest",
    "aligned_row",
    "aligned_column",
    "touching",
    "contains",
    "relative_direction",
    "border_relation",
)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    attrs: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "attrs": self.attrs}


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    predicates: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id, "predicates": self.predicates}


@dataclass(frozen=True)
class ObjectGraph:
    parse_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    nearest_tie_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "parse_id": self.parse_id,
            "nodes": [node.to_json() for node in self.nodes],
            "edges": [edge.to_json() for edge in self.edges],
            "nearest_tie_count": self.nearest_tie_count,
            "tie_policy": "nearest ties remain represented as ties; no target is silently chosen",
        }


def _bbox_center(attrs: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = attrs["bounding_box"]
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _direction(source: dict[str, Any], target: dict[str, Any]) -> str:
    sx, sy = _bbox_center(source)
    tx, ty = _bbox_center(target)
    dx = tx - sx
    dy = ty - sy
    horiz = "east" if dx > 0 else "west" if dx < 0 else "same_col"
    vert = "south" if dy > 0 else "north" if dy < 0 else "same_row"
    return f"{vert}_{horiz}"


def _touching(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_cells = {tuple(cell) for cell in left["cells"]}
    right_cells = {tuple(cell) for cell in right["cells"]}
    return any(abs(x - rx) + abs(y - ry) == 1 for x, y in left_cells for rx, ry in right_cells)


def _contains(left: dict[str, Any], right: dict[str, Any]) -> bool:
    ax0, ay0, ax1, ay1 = left["bounding_box"]
    bx0, by0, bx1, by1 = right["bounding_box"]
    return ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1


def build_object_graph(parse: ParseState) -> ObjectGraph:
    nodes = tuple(GraphNode(obj["object_id"], obj) for obj in parse.objects)
    nearest_targets: dict[str, set[str]] = {}
    nearest_tie_count = 0
    for source in nodes:
        distances = []
        sx, sy = _bbox_center(source.attrs)
        for target in nodes:
            if target.node_id == source.node_id:
                continue
            tx, ty = _bbox_center(target.attrs)
            distances.append((abs(tx - sx) + abs(ty - sy), target.node_id))
        if distances:
            best = min(distance for distance, _ in distances)
            tied = {target_id for distance, target_id in distances if distance == best}
            nearest_targets[source.node_id] = tied
            if len(tied) > 1:
                nearest_tie_count += 1
    edges = []
    for source in nodes:
        for target in nodes:
            if source.node_id == target.node_id:
                continue
            s = source.attrs
            t = target.attrs
            predicates = {
                "same_color": s["color"] == t["color"],
                "same_shape": s["shape_hash"] == t["shape_hash"],
                "same_size": s["cell_count"] == t["cell_count"],
                "nearest": target.node_id in nearest_targets.get(source.node_id, set()),
                "nearest_is_tied": len(nearest_targets.get(source.node_id, set())) > 1,
                "aligned_row": s["top_left_position"][1] == t["top_left_position"][1],
                "aligned_column": s["top_left_position"][0] == t["top_left_position"][0],
                "touching": _touching(s, t),
                "contains": _contains(s, t),
                "relative_direction": _direction(s, t),
                "border_relation": f"{s['border_touch_flag']}->{t['border_touch_flag']}",
            }
            edges.append(GraphEdge(source.node_id, target.node_id, predicates))
    return ObjectGraph(parse.parse_id, nodes, tuple(sorted(edges, key=lambda edge: (edge.source_id, edge.target_id))), nearest_tie_count)


def graph_signature(graph: ObjectGraph) -> str:
    from .r5a3b_parse_hypotheses import stable_hash

    return stable_hash(graph.to_json())
