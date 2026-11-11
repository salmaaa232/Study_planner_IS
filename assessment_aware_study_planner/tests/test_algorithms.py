"""Normal, edge and impossible cases, plus small exhaustive oracles."""
import itertools
import unittest
import numpy as np
from csp.backtracking import solve
from csp.problem import compatible, make_domains, Stats
from csp.ac3 import ac3
from csp.forward_checking import forward_check
from search.core import search
from integration.planner import plan, deadline_feasible
from data.features import FEATURES, with_sessions
from nn.model import MarkNetwork
from nn.baseline import AverageMark


class FixturePredictor:
    """SYNTHETIC fixture only; never used for real evaluation."""
    def predict(self, x):
        x = np.atleast_2d(x)
        return np.clip(x[:, 12] + x[:, [0, 3, 6, 9]].sum(axis=1) / 8, 0, 100)


def fixture_profile(target=55, capacity=2, deadlines=(2, 3, 3)):
    features = [0.] * len(FEATURES)
    features[12] = 40.
    return dict(profile_id='synthetic_fixture', source='SYNTHETIC TEST', target=target, capacity=capacity,
                assignments=[dict(name=f'A{i}', features=list(features), deadline_week=d)
                             for i, d in enumerate(deadlines)])


class NormalCases(unittest.TestCase):
    def test_ucs_known_optimum(self):
        r = search([[40, 50, 60], [30, 45, 60]], [55, 55], 5, lambda s: True)
        self.assertEqual(r.state, (2, 2))
        self.assertGreaterEqual(r.generated, r.expanded)

    def test_all_search_algorithms(self):
        for algorithm in ['ucs', 'greedy', 'astar']:
            r = search([[20, 70, 80], [30, 40, 80]], [60, 60], 5, lambda s: True, algorithm)
            self.assertEqual(sum(r.state), 3)

    def test_all_csp_switches(self):
        for fc, ac in itertools.product([False, True], repeat=2):
            result = solve((2, 1, 1), (1, 2, 2), 2, fc, ac)
            self.assertIsNotNone(result.solution)
            self.assertEqual(len(set(result.solution.values())), 4)
            for x, y in itertools.combinations(result.solution, 2):
                self.assertTrue(compatible(x, result.solution[x], y, result.solution[y]))

    def test_forward_check_removes_slot(self):
        domains = make_domains((1, 1), (1, 1), 2)
        stats = Stats()
        self.assertTrue(forward_check(domains, (0, 0), (1, 1), {(0, 0): (1, 1)}, stats))
        self.assertEqual(domains[(1, 0)], [(1, 2)])
        self.assertEqual(stats.pruned, 1)

    def test_ac3_prunes_order(self):
        domains = make_domains((2,), (1,), 2)
        self.assertTrue(ac3(domains, Stats()))
        self.assertEqual(domains, {(0, 0): [(1, 1)], (0, 1): [(1, 2)]})

    def test_integration(self):
        r = plan(fixture_profile(capacity=3), FixturePredictor())
        self.assertEqual(r['status'], 'solution')
        self.assertEqual(r['allocation'], (3, 3, 3))
        self.assertEqual(len(r['timetable']), 9)

    def test_neural_network_roundtrip_and_monotonicity(self):
        import tempfile
        from pathlib import Path
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 100, (64, 19))
        y = 30 + np.log1p(x[:, :12]).sum(axis=1)
        m = MarkNetwork(seed=1).fit(x, y, epochs=8)
        more = np.stack([with_sessions(row, 3) for row in x])
        self.assertTrue(np.all(m.predict(more) >= m.predict(x) - 1e-8))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'm.npz'
            m.save(p)
            np.testing.assert_allclose(m.predict(x), MarkNetwork.load(p).predict(x))


class EdgeCases(unittest.TestCase):
    def test_empty_problem(self):
        self.assertEqual(solve((), (), 0).solution, {})
        self.assertEqual(search([], [], 0, lambda s: True).state, ())

    def test_target_already_met_zero_capacity(self):
        r = plan(fixture_profile(target=30, capacity=0), FixturePredictor())
        self.assertEqual(r['allocation'], (0, 0, 0))
        self.assertEqual(r['status'], 'solution')

    def test_exact_deadline(self):
        r = solve((2,), (1,), 2)
        self.assertEqual(set(r.solution.values()), {(1, 1), (1, 2)})

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            solve((-1,), (1,), 2)
        with self.assertRaises(ValueError):
            search([[float('nan')]], [55], 1, lambda s: True)
        with self.assertRaises(ValueError):
            plan(fixture_profile(target=101), FixturePredictor())

    def test_limit_not_reported_as_impossible(self):
        r = search([[0, 80]], [55], 1, lambda s: True, max_nodes=0)
        self.assertIn('unknown', r.status)

    def test_nonmonotonic_curves(self):
        for algo in ['ucs', 'astar']:
            r = search([[10, 80, 20, 90], [10, 20, 70]], [75, 65], 4, lambda s: True, algo)
            self.assertEqual(r.state, (1, 2))

    def test_baseline_is_effort_independent(self):
        x = np.zeros((1, 19)); x[0, 12] = 61
        baseline = AverageMark(70)
        self.assertEqual(baseline.predict(x)[0], 61)
        self.assertEqual(baseline.predict(with_sessions(x[0], 4))[0], 61)
        x[0, 14] = 1
        self.assertEqual(baseline.predict(x)[0], 70)


class ImpossibleCases(unittest.TestCase):
    def test_deadline_conflict_all_variants(self):
        for fc, ac in itertools.product([False, True], repeat=2):
            r = solve((2, 1), (1, 2), 1, fc, ac)
            self.assertIsNone(r.solution)
            self.assertIn('no solution', r.message)

    def test_unreachable_marks_and_best_effort(self):
        r = plan(fixture_profile(target=99, capacity=1), FixturePredictor(), max_sessions=2)
        self.assertEqual(r['status'], 'no solution')
        self.assertTrue(r['best_effort'])
        self.assertLessEqual(sum(r['allocation']), 3)
        self.assertEqual(len(r['timetable']), sum(r['allocation']))

    def test_predictor_accepts_csp_rejects(self):
        r = plan(fixture_profile(capacity=3, deadlines=(0, 3, 3)), FixturePredictor(), max_sessions=3)
        self.assertEqual(r['status'], 'no solution')
        self.assertGreater(r['search']['csp_rejections'], 0)

    def test_zero_deadline(self):
        self.assertIsNone(solve((1,), (0,), 3).solution)


class ExhaustiveOracles(unittest.TestCase):
    def test_csp_matches_bruteforce(self):
        for counts in itertools.product(range(3), repeat=2):
            for deadlines in itertools.product(range(3), repeat=2):
                domains = make_domains(counts, deadlines, 1)
                variables = list(domains)
                brute = any(all(compatible(variables[i], values[i], variables[j], values[j])
                                for i in range(len(variables)) for j in range(i))
                            for values in itertools.product(*domains.values()))
                self.assertEqual(deadline_feasible(counts, deadlines, 1), brute)
                for fc, ac in itertools.product([False, True], repeat=2):
                    self.assertEqual(solve(counts, deadlines, 1, fc, ac).solution is not None, brute)

    def test_search_matches_exhaustive_minimum(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            curves = rng.integers(20, 90, (3, 4)).tolist()
            feasible = lambda s: deadline_feasible(s, [1, 2, 3], 2)
            good = [s for s in itertools.product(range(4), repeat=3)
                    if sum(s) <= 6 and feasible(s) and all(curves[a][s[a]] >= 55 for a in range(3))]
            expected = min(map(sum, good)) if good else None
            for algo in ['ucs', 'astar']:
                r = search(curves, [55]*3, 6, feasible, algo)
                self.assertEqual(sum(r.state) if r.state is not None else None, expected)


if __name__ == '__main__':
    unittest.main()
