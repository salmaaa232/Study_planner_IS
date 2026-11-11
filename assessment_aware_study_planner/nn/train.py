"""Student-grouped validation; later-presentation test evaluated after fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import FEATURES, with_sessions
from nn.model import MarkNetwork
from nn.baseline import AverageMark

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, pred):
    return dict(mae=float(np.mean(np.abs(y - pred))), rmse=float(np.sqrt(np.mean((y - pred) ** 2))))


def train(data_dir, output_dir, seed=42, epochs=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(Path(data_dir) / 'train.csv')
    te = pd.read_csv(Path(data_dir) / 'test.csv')
    metadata = json.loads((Path(data_dir) / 'metadata.json').read_text())
    ids = np.array(sorted(tr.id_student.unique()))
    np.random.default_rng(seed).shuffle(ids)
    validation_ids = set(ids[:max(1, int(.2 * len(ids)))])
    val_mask = tr.id_student.isin(validation_ids)
    fit, val = tr[~val_mask], tr[val_mask]
    if fit.empty or val.empty or te.empty:
        raise ValueError('Need nonempty training, validation and test data')
    network = MarkNetwork(seed=seed).fit(fit[FEATURES].values, fit.score.values, epochs=epochs,
                        validation=(val[FEATURES].values, val.score.values))
    network.save(output_dir / 'model.npz')
    baseline = AverageMark(fit.score.mean())
    x, y = te[FEATURES].values, te.score.values
    # Check all test rows along session increments 0..8.
    previous = network.predict(x)
    violations, comparisons = 0, 0
    for sessions in range(1, 9):
        more = np.stack([with_sessions(row, sessions) for row in x])
        current = network.predict(more)
        violations += int(np.sum(current < previous - 1e-8))
        comparisons += len(x)
        previous = current
    result = dict(seed=seed, source=metadata['source'], fit_rows=len(fit), validation_rows=len(val),
                  test_rows=len(te), nn=metrics(y, network.predict(x)),
                  baseline=metrics(y, baseline.predict(x)), baseline_training_mean=baseline.training_mean,
                  monotonicity_violations=violations, monotonicity_comparisons=comparisons,
                  epochs=len(network.history), train_presentations=metadata['train_presentations'],
                  test_presentation=metadata['test_presentation'], feature_names=FEATURES,
                  training_click_max=fit[FEATURES[:12]].max().tolist())
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    (output_dir / 'learning_curve.json').write_text(json.dumps(network.history, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return network, baseline, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    a = p.parse_args()
    train(a.data_dir, a.output_dir, a.seed, a.epochs)


if __name__ == '__main__':
    main()
