"""Produce allocations, timetables and marks for all 30 real profiles."""
import json
from pathlib import Path
import pandas as pd
from nn.model import MarkNetwork
from integration.planner import plan


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/profiles'
    out.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((root / 'data/processed/profiles.json').read_text())
    model = MarkNetwork.load(root / 'models/model.npz')
    rows = []
    for profile in profiles:
        result = plan(profile, model)
        (out / f"{profile['profile_id']}.json").write_text(json.dumps(result, indent=2))
        rows.append(dict(profile_id=profile['profile_id'], status=result['status'],
                         target=profile['target'], allocation=json.dumps(result['allocation']),
                         predicted_marks=json.dumps(result['predicted_marks']),
                         total_sessions=sum(result['allocation']),
                         expanded=result['search']['expanded'], timetable_file=f"{profile['profile_id']}.json"))
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    print(f'Wrote {len(rows)} profiles to {out}')


if __name__ == '__main__':
    main()
