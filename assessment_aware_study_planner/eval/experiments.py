"""Three paired experiments, five independent seeds, raw rows and mean +/- SD."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn.train import train
from nn.model import MarkNetwork
from nn.baseline import AverageMark
from integration.planner import plan
from csp.backtracking import solve
from data.features import with_sessions

ROOT = Path(__file__).resolve().parents[1]


def summarize(frame, group, metrics, out):
    # Each seed is an experimental unit; first average cases within that seed.
    per_seed = frame.groupby(['seed'] + group, dropna=False)[metrics].mean().reset_index()
    per_seed.to_csv(out.with_name(out.stem + '_per_seed.csv'), index=False)
    summary = per_seed.groupby(group, dropna=False)[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out, index=False)
    return summary.reset_index()


def run(seeds, out, epochs=100):
    out.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((ROOT / 'data/processed/profiles.json').read_text())
    search_rows, csp_rows, predictor_rows, accuracy_rows = [], [], [], []
    for seed in seeds:
        print(f'Experiment seed {seed}', flush=True)
        model_dir = out / f'model_seed_{seed}'
        if (model_dir / 'metrics.json').exists():
            metrics = json.loads((model_dir / 'metrics.json').read_text())
            network = MarkNetwork.load(model_dir / 'model.npz')
            baseline = AverageMark(metrics['baseline_training_mean'])
        else:
            network, baseline, metrics = train(ROOT / 'data/processed', model_dir, seed, epochs)
        for name in ['nn', 'baseline']:
            accuracy_rows.append(dict(seed=seed, predictor=name, **metrics[name],
                                      monotonicity_violations=metrics['monotonicity_violations'] if name == 'nn' else 0))
        rng = np.random.default_rng(seed)
        for index, original in enumerate(profiles):
            profile = copy.deepcopy(original)
            profile['target'] = float(rng.choice([55, 60, 65, 75, 90]))
            # Controlled effort thresholds exercise nontrivial search. These
            # generated targets are NOT evidence of educational effectiveness.
            challenge = copy.deepcopy(profile)
            for task in challenge['assignments']:
                required = int(rng.integers(1, 5))
                task['target'] = float(network.predict(with_sessions(task['features'], required))[0])
            for algorithm in ['ucs', 'greedy', 'astar']:
                started = time.perf_counter()
                result = plan(challenge, network, algorithm, include_fallback=False)
                search_rows.append(dict(seed=seed, case=index, algorithm=algorithm,
                                        scenario='generated_attainable_targets',
                                        targets=json.dumps([t['target'] for t in challenge['assignments']]),
                                        expanded=result['search']['expanded'], generated=result['search']['generated'],
                                        csp_rejections=result['search']['csp_rejections'],
                                        model_evaluations=result['model_evaluations'],
                                        success=int(result['status'] == 'solution'),
                                        cost=sum(result['allocation']) if result['status'] == 'solution' else np.nan,
                                        seconds=time.perf_counter()-started))
            for name, predictor in [('nn', network), ('baseline', baseline)]:
                result = plan(profile, predictor, 'ucs', include_fallback=False)
                predictor_rows.append(dict(seed=seed, case=index, predictor=name, target=profile['target'],
                                           success=int(result['status'] == 'solution'),
                                           cost=sum(result['allocation']) if result['status'] == 'solution' else np.nan,
                                           expanded=result['search']['expanded']))
        for case in range(12):
            # Generated scheduling stress tests, independent of the NN.
            counts = rng.integers(1, 4, size=3).tolist()
            while sum(counts) > 7:
                counts[int(np.argmax(counts))] -= 1
            deadlines = rng.permutation([1, 2, 3]).tolist()
            for capacity in [2, 3, 4]:
                for name, fc, ac in [('BT', False, False), ('BT+FC', True, False), ('BT+FC+AC3', True, True)]:
                    started = time.perf_counter()
                    result = solve(tuple(counts), tuple(deadlines), capacity, fc, ac)
                    csp_rows.append(dict(seed=seed, case=case, capacity=capacity, mode=name,
                                         allocation=json.dumps(counts), deadlines=json.dumps(deadlines),
                                         success=int(result.solution is not None), assignments=result.stats.assignments,
                                         backtracks=result.stats.backtracks, pruned=result.stats.pruned,
                                         checks=result.stats.checks, arcs=result.stats.arcs,
                                         seconds=time.perf_counter()-started))
        # Checkpoint after each seed. Completed cached models avoid repeated fits.
        for name, rows in [('search', search_rows), ('csp', csp_rows), ('predictor', predictor_rows), ('accuracy', accuracy_rows)]:
            pd.DataFrame(rows).to_csv(out / f'{name}_raw.csv', index=False)
    search_summary = summarize(pd.DataFrame(search_rows), ['algorithm'],
                              ['expanded', 'generated', 'cost', 'success', 'seconds'], out / 'search_summary.csv')
    csp_summary = summarize(pd.DataFrame(csp_rows), ['capacity', 'mode'],
                           ['assignments', 'backtracks', 'pruned', 'checks', 'seconds'], out / 'csp_summary.csv')
    predictor_summary = summarize(pd.DataFrame(predictor_rows), ['predictor'],
                                 ['success', 'cost', 'expanded'], out / 'predictor_summary.csv')
    accuracy_summary = summarize(pd.DataFrame(accuracy_rows), ['predictor'],
                                ['mae', 'rmse', 'monotonicity_violations'], out / 'accuracy_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['expanded', 'cost'], ['Nodes expanded', 'Sessions in successful plans']):
        ax.bar(search_summary.algorithm, search_summary[f'{metric}_mean'], yerr=search_summary[f'{metric}_std'], capsize=5)
        ax.set(title=title, ylabel='Mean across seed means')
    fig.suptitle('Experiment 1: UCS, Greedy and A* | error bars = sample SD across seeds')
    fig.savefig(out / 'experiment1_search.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for mode in csp_summary['mode'].unique():
        subset = csp_summary[csp_summary['mode'] == mode]
        for ax, metric in zip(axes, ['assignments', 'backtracks']):
            ax.errorbar(subset.capacity, subset[f'{metric}_mean'], yerr=subset[f'{metric}_std'], marker='o', capsize=4, label=mode)
            ax.set(xlabel='Slots per week', ylabel=metric.title(), xticks=[2, 3, 4])
    axes[0].legend()
    fig.suptitle('Experiment 2: propagation on generated scheduling cases | mean and SD')
    fig.savefig(out / 'experiment2_csp.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].bar(accuracy_summary.predictor, accuracy_summary.mae_mean, yerr=accuracy_summary.mae_std, capsize=5)
    axes[0].set(title='Later-presentation prediction error', ylabel='MAE (mark points)')
    axes[1].bar(predictor_summary.predictor, predictor_summary.success_mean, yerr=predictor_summary.success_std, capsize=5)
    axes[1].set(title='Plans meeting their own model targets', ylabel='Fraction', ylim=(0, 1))
    fig.suptitle('Experiment 3: NN versus average baseline | feasibility is not educational benefit')
    fig.savefig(out / 'experiment3_predictor.png', dpi=160); plt.close(fig)
    print('Completed all three experiments:', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/experiments')
    p.add_argument('--epochs', type=int, default=100)
    a = p.parse_args()
    if len(set(a.seeds)) < 5:
        p.error('Use at least five distinct seeds for the required experiments')
    run(a.seeds, a.output_dir, a.epochs)


if __name__ == '__main__':
    main()
