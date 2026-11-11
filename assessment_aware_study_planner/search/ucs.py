from .core import search

def ucs(curves, targets, max_total, feasible, **kwargs):
    return search(curves, targets, max_total, feasible, algorithm='ucs', **kwargs)
