"""Binary CSP: session variables, (week, slot) values, deadline and order rules."""
from dataclasses import dataclass, field


@dataclass
class Stats:
    assignments: int = 0  # consistent value attempts
    backtracks: int = 0  # exhausted recursive calls
    checks: int = 0
    pruned: int = 0  # individual domain values removed
    arcs: int = 0  # AC-3 arcs processed
    trace: list = field(default_factory=list)

    def event(self, kind, **kwargs):
        if len(self.trace) < 150:
            self.trace.append(dict(kind=kind, **kwargs))


def compatible(x, vx, y, vy):
    if vx == vy:
        return False
    if x[0] == y[0]:
        return vx < vy if x[1] < y[1] else vx > vy
    return True


def make_domains(allocation, deadlines, capacity):
    if len(allocation) != len(deadlines):
        raise ValueError('One deadline is required per assignment')
    if any(not isinstance(v, int) or v < 0 for v in [*allocation, *deadlines, capacity]):
        raise ValueError('Allocation, deadlines and capacity must be nonnegative integers')
    return {(a, s): [(w, k) for w in range(1, deadlines[a] + 1)
                      for k in range(1, capacity + 1)]
            for a, count in enumerate(allocation) for s in range(count)}
