"""Dependency graph resolution using topological sort."""

from __future__ import annotations

from collections import deque


def topological_sort(
    nodes: list[str],
    dependencies: dict[str, list[str]],
) -> list[str]:
    """Topological sort of nodes respecting dependency order.

    Returns nodes in dependency order (prerequisites first).
    Raises ValueError if a cycle is detected.
    """
    valid_nodes: set[str] = set(nodes)
    in_degree: dict[str, int] = {node: 0 for node in nodes}
    adj: dict[str, list[str]] = {node: [] for node in nodes}

    for node in nodes:
        for dep in dependencies.get(node, []):
            if dep in valid_nodes:
                adj[dep].append(node)
                in_degree[node] += 1

    queue: deque[str] = deque([node for node in nodes if in_degree[node] == 0])
    sorted_nodes: list[str] = []

    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Cycle detected in dependency graph")

    return sorted_nodes
