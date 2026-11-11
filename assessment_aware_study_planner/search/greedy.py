from .core import search

def greedy(curves, targets, max_total, feasible, **kwargs):
    return search(curves, targets, max_total, feasible, algorithm='greedy', **kwargs)
