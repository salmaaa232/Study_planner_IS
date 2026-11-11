"""Finite graph search over session-count tuples; every edge costs one session."""
import heapq
import itertools
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SearchResult:
    state: tuple | None
    status: str
    expanded: int = 0
    generated: int = 1  # includes root; unique states enqueued
    csp_rejections: int = 0
    trace: list = field(default_factory=list)


def search(curves, targets, max_total, feasible, algorithm='ucs', max_nodes=None):
    """Curves[a][s] is predicted score for assignment a given s sessions.

    h(s) sums the minimum *additional* sessions needed to hit each target.
    It ignores scheduling and is an admissible, consistent lower bound,
    including for nonmonotone finite score curves. Precomputation is counted
    separately by the planner; it is not hidden NN work inside node counters.
    """
    if algorithm not in {'ucs', 'greedy', 'astar'}:
        raise ValueError('algorithm must be ucs, greedy or astar')
    if not isinstance(max_total, int) or max_total < 0:
        raise ValueError('max_total must be a nonnegative integer')
    if len(curves) != len(targets) or any(len(c) == 0 for c in curves):
        raise ValueError('One nonempty prediction curve is required per target')
    if any(not np.all(np.isfinite(c)) for c in curves) or not np.all(np.isfinite(targets)):
        raise ValueError('Predictions and targets must be finite')
    root = (0,) * len(curves)
    result = SearchResult(None, 'no solution')

    def heuristic(state):
        h = 0
        for a, s in enumerate(state):
            candidates = [k - s for k in range(s, len(curves[a])) if curves[a][k] >= targets[a]]
            if not candidates:
                return float('inf')
            h += min(candidates)
        return h

    def priority(state):
        if algorithm == 'ucs':
            return sum(state)
        h = heuristic(state)
        return h if algorithm == 'greedy' else sum(state) + h

    sequence = itertools.count()
    frontier = [(priority(root), next(sequence), root)]
    seen = {root}
    while frontier:
        if max_nodes is not None and result.expanded >= max_nodes:
            result.status = 'search limit reached; feasibility unknown'
            return result
        _, _, state = heapq.heappop(frontier)
        result.expanded += 1
        marks = [float(curves[a][s]) for a, s in enumerate(state)]
        meets = all(p >= t for p, t in zip(marks, targets))
        schedulable = feasible(state) if meets else None
        if len(result.trace) < 250:
            result.trace.append(dict(state=state, cost=sum(state), marks=marks,
                                     meets_targets=meets, schedulable=schedulable))
        if meets and schedulable:
            result.state, result.status = state, 'solution'
            return result
        if meets:
            result.csp_rejections += 1
            # Adding sessions cannot repair a violated scheduling constraint.
            continue
        if sum(state) >= max_total:
            continue
        for a in range(len(state)):
            if state[a] + 1 >= len(curves[a]):
                continue
            child = list(state)
            child[a] += 1
            child = tuple(child)
            if child not in seen:
                seen.add(child)
                heapq.heappush(frontier, (priority(child), next(sequence), child))
                result.generated += 1
    return result
