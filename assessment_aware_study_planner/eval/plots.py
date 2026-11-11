"""Headless figures for the integrated demo."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2878ad', '#db7b2b', '#36967d']


def demo_plot(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for a, curve in enumerate(result['curves']):
        axes[0, 0].plot(range(len(curve)), curve, marker='.', color=COLORS[a % 3], label=f'A{a+1}')
        axes[0, 0].axhline(result['targets'][a], color=COLORS[a % 3], alpha=.25, linestyle='--')
    axes[0, 0].set(title='1. NN predicts marks for candidate effort', xlabel='Additional sessions', ylabel='Predicted mark')
    axes[0, 0].legend()
    trace = result['search']['trace']
    axes[0, 1].plot([t['cost'] for t in trace], color='#2878ad')
    axes[0, 1].set(title='2. Search explores allocations (first 250)', xlabel='Expanded state', ylabel='Total sessions')
    for item in result['timetable']:
        axes[1, 0].scatter(item['week'], item['slot'], color=COLORS[(item['assignment']-1) % 3], s=500, marker='s')
        axes[1, 0].annotate(f"A{item['assignment']}.{item['session']}", (item['week'], item['slot']),
                          ha='center', va='center', color='white', fontsize=8)
    if not result['timetable']:
        axes[1, 0].text(.5, .5, 'No additional sessions scheduled', ha='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set(title='3. CSP assigns valid weekly slots', xlabel='Week after decision date', ylabel='Slot')
    axes[1, 0].set_yticks(range(1, result['capacity']+1))
    last_week = max((s['week'] for s in result['timetable']), default=1)
    axes[1, 0].set_xticks(range(1, last_week + 1))
    axes[1, 0].set_xlim(.5, last_week + .5)
    axes[1, 0].set_ylim(.5, max(1, result['capacity']) + .5)
    labels = [f'A{i+1}' for i in range(len(result['targets']))]
    if result['predicted_marks'] is not None:
        axes[1, 1].bar(labels, result['predicted_marks'], color=COLORS[:len(labels)])
    axes[1, 1].scatter(labels, result['targets'], color='#9b2226', marker='_', s=400, label='Target')
    axes[1, 1].set(title='Result: ' + result['status'], ylabel='Predicted mark', ylim=(0, 100))
    axes[1, 1].legend()
    fig.suptitle('Assessment-Aware Study Planner | observational what-if estimates', fontsize=14)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
