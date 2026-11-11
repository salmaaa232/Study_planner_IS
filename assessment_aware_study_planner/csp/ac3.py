"""AC-3 preprocessing, independently switchable from forward checking."""
from collections import deque
from .problem import compatible


def ac3(domains, stats):
    queue = deque((x, y) for x in domains for y in domains if x != y)
    while queue:
        x, y = queue.popleft()
        stats.arcs += 1
        kept = []
        for vx in domains[x]:
            supported = False
            for vy in domains[y]:
                stats.checks += 1
                if compatible(x, vx, y, vy):
                    supported = True
                    break
            if supported:
                kept.append(vx)
        if len(kept) < len(domains[x]):
            stats.event('AC3', variable=list(x), before=len(domains[x]), after=len(kept))
            stats.pruned += len(domains[x]) - len(kept)
            domains[x] = kept
            if not kept:
                return False
            queue.extend((z, x) for z in domains if z != x and z != y)
    return all(domains.values())
