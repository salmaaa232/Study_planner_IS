"""Run the trained NN -> search -> CSP pipeline, with JSON trace and PNG."""
import argparse
import copy
import json
from pathlib import Path
from nn.model import MarkNetwork
from nn.baseline import AverageMark
from integration.planner import plan
from eval.plots import demo_plot
from csp.backtracking import solve
from data.features import with_sessions

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=int, default=5, help='Zero-based profile number, 0..29')
    p.add_argument('--target', type=float, default=60)
    p.add_argument('--capacity', type=int, default=3)
    p.add_argument('--algorithm', choices=['ucs', 'greedy', 'astar'], default='ucs')
    p.add_argument('--csp', choices=['bt', 'fc', 'ac3'], default='ac3')
    p.add_argument('--predictor', choices=['nn', 'baseline'], default='nn')
    p.add_argument('--max-sessions', type=int, default=8)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--rejection-demo', action='store_true')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/demo')
    args = p.parse_args()
    if not (ROOT / 'models/model.npz').exists():
        p.error('Train first: python data/make_data.py then python nn/train.py')
    profiles = json.loads((ROOT / 'data/processed/profiles.json').read_text())
    if not 0 <= args.profile < len(profiles):
        p.error(f'--profile must be between 0 and {len(profiles)-1}')
    profile = copy.deepcopy(profiles[args.profile])
    profile.update(target=args.target, capacity=args.capacity)
    model = MarkNetwork.load(ROOT / 'models/model.npz')
    metrics = json.loads((ROOT / 'models/metrics.json').read_text())
    predictor = model if args.predictor == 'nn' else AverageMark(metrics['baseline_training_mean'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('OULAD REAL PROFILE; generated capacity, target and session-to-click assumptions.')
    print('Predictions are observational what-if estimates, not promised marks.')
    print(json.dumps(dict(profile_id=profile['profile_id'], target=profile['target'], capacity=profile['capacity'],
                          deadlines=[a['deadline_week'] for a in profile['assignments']], predictor=args.predictor), indent=2))
    result = plan(profile, predictor, args.algorithm, args.csp != 'bt', args.csp == 'ac3', args.max_sessions)
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2))
    demo_plot(result, args.output_dir / 'cooperation.png')
    print(result['status'].upper())
    print('Allocation:', result['allocation'], '| Predicted marks:', result['predicted_marks'])
    if result['best_effort']:
        print('Best feasible compromise minimizes total target shortfall within the session cap.')
    print('Week  Slot  Assignment  Session')
    for s in result['timetable']:
        print(f"{s['week']:4}  {s['slot']:4}  {s['assignment']:10}  {s['session']:7}")
    print('Search:', {k: result['search'][k] for k in ['expanded', 'generated', 'csp_rejections']})
    print('CSP:', result['csp_counters'])
    if args.trace:
        print('SEARCH TRACE (up to first 30 states)')
        print(json.dumps(result['search']['trace'][:30], indent=2))
        print('CSP TRACE (up to first 30 events)')
        print(json.dumps(result['csp_trace'][:30], indent=2))
    if args.rejection_demo:
        # A concrete NN-approved allocation rejected for scheduling reasons.
        allocation = (2, 1, 1)
        marks = [float(model.predict(with_sessions(a['features'], n))[0])
                 for a, n in zip(profile['assignments'], allocation)]
        targets = [m - .01 for m in marks]
        comparison = []
        print('\nGENERATED REJECTION SCENARIO: NN marks meet constructed targets; A1 needs 2 sessions in 1 slot.')
        print('Allocation:', allocation, 'marks:', marks, 'targets:', targets, 'deadlines: [1, 2, 3], capacity: 1')
        for name, fc, ac in [('BT', False, False), ('BT+FC', True, False), ('BT+FC+AC3', True, True)]:
            csp = solve(allocation, (1, 2, 3), 1, fc, ac)
            item = dict(mode=name, message=csp.message, assignments=csp.stats.assignments,
                        backtracks=csp.stats.backtracks, pruned=csp.stats.pruned, trace=csp.stats.trace)
            comparison.append(item)
            print(json.dumps(item, indent=2))
        (args.output_dir / 'rejection.json').write_text(json.dumps(dict(allocation=allocation, marks=marks,
                               targets=targets, deadlines=[1, 2, 3], capacity=1, variants=comparison), indent=2))
    print('Saved:', args.output_dir / 'result.json', 'and cooperation.png')


if __name__ == '__main__':
    main()
