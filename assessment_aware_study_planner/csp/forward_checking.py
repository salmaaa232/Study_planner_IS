"""Prune every unassigned neighbor after choosing one value."""
from .problem import compatible


def forward_check(domains, variable, value, assigned, stats):
    for other in domains:
        if other in assigned:
            continue
        old = domains[other]
        kept = []
        for candidate in old:
            stats.checks += 1
            if compatible(variable, value, other, candidate):
                kept.append(candidate)
        stats.pruned += len(old) - len(kept)
        domains[other] = kept
        if len(kept) != len(old):
            stats.event('FC', variable=list(other), before=len(old), after=len(kept))
        if not kept:
            return False
    return True
