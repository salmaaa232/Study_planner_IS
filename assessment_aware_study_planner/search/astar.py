from .core import search

def astar(curves, targets, max_total, feasible, **kwargs):
    return search(curves, targets, max_total, feasible, algorithm='astar', **kwargs)
