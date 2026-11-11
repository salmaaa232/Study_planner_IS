"""Plain BT, BT+FC, BT+AC3, or BT+FC+AC3 with identical variable order."""
from dataclasses import dataclass
from .problem import Stats, make_domains, compatible
from .forward_checking import forward_check
from .ac3 import ac3


@dataclass
class CSPResult:
    solution: dict | None
    stats: Stats

    @property
    def message(self):
        return 'solution' if self.solution is not None else 'no solution: deadline/capacity conflict'


def solve(allocation, deadlines, capacity, use_fc=True, use_ac3=True):
    domains = make_domains(allocation, deadlines, capacity)
    stats = Stats()
    if any(not d for d in domains.values()):
        return CSPResult(None, stats)
    if use_ac3 and not ac3(domains, stats):
        return CSPResult(None, stats)
    variables = list(domains)  # deliberately fixed for fair propagation comparisons

    def visit(assigned, current):
        if len(assigned) == len(variables):
            return dict(assigned)
        x = variables[len(assigned)]
        for value in current[x]:
            ok = True
            for y, vy in assigned.items():
                stats.checks += 1
                if not compatible(x, value, y, vy):
                    ok = False
                    break
            if not ok:
                continue
            stats.assignments += 1
            stats.event('assign', variable=list(x), value=list(value))
            assigned[x] = value
            child = {v: list(d) for v, d in current.items()}
            child[x] = [value]
            if not use_fc or forward_check(child, x, value, assigned, stats):
                result = visit(assigned, child)
                if result is not None:
                    return result
            del assigned[x]
        stats.backtracks += 1
        return None

    return CSPResult(visit({}, domains), stats)


def timetable(solution):
    if solution is None:
        return []
    return [dict(assignment=a + 1, session=s + 1, week=w, slot=k)
            for (a, s), (w, k) in sorted(solution.items(), key=lambda item: item[1])]
