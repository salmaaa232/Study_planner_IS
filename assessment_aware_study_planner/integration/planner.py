"""Search proposes allocations, predictor checks marks, CSP checks timing."""
from dataclasses import asdict
import itertools
import numpy as np
from data.features import with_sessions
from search.core import search
from csp.backtracking import solve, timetable


def deadline_feasible(state, deadlines, capacity):
    """Exact feasibility check for our prefix-slot domains, used for fallback.

    Every task is available from week 1; earliest-deadline-first is feasible
    iff the cumulative demand by every deadline fits the available slots.
    The main search still calls the actual CSP to demonstrate cooperation.
    """
    return all(sum(s for s, d in zip(state, deadlines) if d <= deadline) <= deadline * capacity
               for deadline in set(deadlines))


def best_effort(curves, targets, deadlines, capacity, max_total):
    """Exact fallback: minimize summed mark shortfall, then session count.
    Report one jointly feasible compromise, not incompatible per-task maxima.
    """
    best = None
    for state in itertools.product(*(range(len(c)) for c in curves)):
        if sum(state) > max_total or not deadline_feasible(state, deadlines, capacity):
            continue
        marks = [float(curves[a][s]) for a, s in enumerate(state)]
        key = (sum(max(0., t - p) for t, p in zip(targets, marks)), sum(state), state)
        if best is None or key < best[0]:
            best = (key, state, marks)
    return best


def plan(profile, predictor, algorithm='ucs', use_fc=True, use_ac3=True, max_sessions=8,
         max_nodes=None, include_fallback=True):
    tasks = profile['assignments']
    capacity = profile['capacity']
    deadlines = [a['deadline_week'] for a in tasks]
    if not isinstance(capacity, int) or capacity < 0 or not isinstance(max_sessions, int) or max_sessions < 0:
        raise ValueError('Capacity and session cap must be nonnegative integers')
    if any(not isinstance(d, int) or d < 0 for d in deadlines):
        raise ValueError('Deadline weeks must be nonnegative integers')
    targets = [a.get('target', profile['target']) for a in tasks]
    if any(not np.isfinite(t) or not 0 <= t <= 100 for t in targets):
        raise ValueError('Targets must be finite and between 0 and 100')
    curves = [predictor.predict(np.stack([with_sessions(a['features'], s)
                                         for s in range(max_sessions + 1)])).tolist() for a in tasks]
    max_total = capacity * max(deadlines, default=0)
    cache = {}

    def feasible(state):
        if state not in cache:
            cache[state] = solve(state, deadlines, capacity, use_fc, use_ac3)
        return cache[state].solution is not None

    result = search(curves, targets, max_total, feasible, algorithm, max_nodes)
    allocation, marks, schedule = None, None, []
    fallback = False
    if result.state is not None:
        allocation = result.state
        marks = [curves[a][s] for a, s in enumerate(allocation)]
        schedule = timetable(cache[allocation].solution)
    elif include_fallback and result.status == 'no solution':
        best = best_effort(curves, targets, deadlines, capacity, max_total)
        if best is not None:
            _, allocation, marks = best
            feasible(allocation)
            schedule = timetable(cache[allocation].solution)
            fallback = True
    return dict(profile_id=profile.get('profile_id', 'custom'), source=profile.get('source', 'custom'),
                status=result.status, allocation=allocation, predicted_marks=marks, timetable=schedule,
                best_effort=fallback, targets=targets, curves=curves, search=asdict(result),
                algorithm=algorithm, capacity=capacity, deadlines=deadlines,
                max_sessions_per_assignment=max_sessions, model_evaluations=len(tasks)*(max_sessions+1),
                scope='No-solution and fallback are exact only within the configured session cap.',
                csp_calls=len(cache),
                csp_counters={name:sum(getattr(r.stats, name) for r in cache.values())
                              for name in ['assignments', 'backtracks', 'checks', 'pruned', 'arcs']},
                csp_trace=[event for r in cache.values() for event in r.stats.trace][:150])
