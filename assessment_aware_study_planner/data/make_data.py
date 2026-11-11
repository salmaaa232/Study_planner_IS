"""Download the pinned OULAD archive, validate it, and preprocess one course."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import FEATURES, activity_group, snapshot

URL = 'https://ndownloader.figshare.com/files/8606371'
MD5 = '7412686fd77cf0e0ee1e8c3e9b354308'
TABLES = ['assessments.csv', 'studentAssessment.csv', 'studentVle.csv', 'vle.csv']
ROOT = Path(__file__).resolve().parents[1]


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'anonymisedData.zip'
    if not archive.exists():
        print('Downloading public OULAD archive (46,750,706 bytes; no login)...', flush=True)
        temporary = archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5:
        raise ValueError('OULAD checksum mismatch; remove the archive and download again')
    with zipfile.ZipFile(archive) as source:
        for name in TABLES:
            member = next((n for n in source.namelist() if Path(n).name == name), None)
            if member is None:
                raise ValueError(f'Missing table: {name}')
            # Write only named CSVs, never extract arbitrary archive paths.
            if not (raw / name).exists():
                with source.open(member) as inp, (raw / name).open('wb') as out:
                    shutil.copyfileobj(inp, out)
    return digest


def preprocess(raw, out, module='BBB', seed=42):
    out.mkdir(parents=True, exist_ok=True)
    assessments = pd.read_csv(raw / 'assessments.csv')
    assessments = assessments[(assessments.code_module == module) &
                              assessments.code_presentation.str.endswith('J') &
                              assessments.assessment_type.isin(['TMA', 'CMA'])].dropna(subset=['date'])
    presentations = sorted(assessments.code_presentation.unique())
    if len(presentations) < 2:
        raise ValueError('Need at least two October presentations for chronological train/test')
    scores = pd.read_csv(raw / 'studentAssessment.csv').merge(assessments, on='id_assessment', validate='many_to_one')
    scores = scores[(scores.is_banked == 0) & scores.score.between(0, 100)].copy()
    scores['available_day'] = scores[['date', 'date_submitted']].max(axis=1) + 14
    sites = pd.read_csv(raw / 'vle.csv')
    sites = sites[sites.code_module == module][['id_site', 'code_module', 'code_presentation', 'activity_type']]
    pieces = []
    for number, chunk in enumerate(pd.read_csv(raw / 'studentVle.csv', chunksize=250_000)):
        chunk = chunk[(chunk.code_module == module) & chunk.code_presentation.isin(presentations)]
        if chunk.empty:
            continue
        chunk = chunk.merge(sites, on=['id_site', 'code_module', 'code_presentation'], how='left', validate='many_to_one')
        if chunk.activity_type.isna().any():
            raise ValueError('Unmapped activity site')
        chunk['group'] = chunk.activity_type.map(activity_group)
        pieces.append(chunk.groupby(['code_presentation', 'id_student', 'date', 'group'], as_index=False).sum_click.sum())
    daily = pd.concat(pieces).groupby(['code_presentation', 'id_student', 'date', 'group'], as_index=False).sum_click.sum()
    click_groups = {key: group for key, group in daily.groupby(['code_presentation', 'id_student'])}
    empty = daily.iloc[:0]
    rows = []
    histories = {key: group for key, group in scores.groupby(['code_presentation', 'id_student'])}
    for key, history in histories.items():
        clicks = click_groups.get(key, empty)
        for assessment in history.to_dict('records'):
            # Include all horizons used by the common-cutoff demo profiles.
            for horizon in [14, 28, 42, 70, 112]:
                cutoff = int(assessment['date'] - horizon)
                if assessment['date_submitted'] <= cutoff:
                    continue
                row = snapshot(clicks, history, cutoff, assessment)
                row.update(code_module=module, code_presentation=key[0], id_student=int(key[1]),
                           id_assessment=int(assessment['id_assessment']), cutoff_day=cutoff,
                           deadline_day=int(assessment['date']), score=float(assessment['score']))
                rows.append(row)
    frame = pd.DataFrame(rows)
    test_presentation = presentations[-1]
    # Remove repeated test students from training to prevent student overlap.
    test_ids = set(frame.loc[frame.code_presentation == test_presentation, 'id_student'])
    train = frame[(frame.code_presentation != test_presentation) & ~frame.id_student.isin(test_ids)]
    test = frame[frame.code_presentation == test_presentation]
    train.to_csv(out / 'train.csv', index=False)
    test.to_csv(out / 'test.csv', index=False)

    # Common decision date for each three-assignment profile. Future marks are
    # never fed into these profiles. Pick assignments 2, 3, 4 chronologically.
    upcoming = assessments[assessments.code_presentation == test_presentation].sort_values(['date', 'id_assessment']).iloc[1:4]
    if len(upcoming) != 3:
        raise ValueError('Need four non-exam assignments to build demonstration profiles')
    cutoff = int(upcoming.date.min() - 14)
    candidates = sorted(k for k in histories if k[0] == test_presentation)
    rng = np.random.default_rng(seed)
    profiles = []
    for index in rng.permutation(len(candidates)):
        key = candidates[index]
        history = histories[key]
        # Exclude any profile already submitted for one of the upcoming tasks.
        early = history[history.id_assessment.isin(upcoming.id_assessment) & (history.date_submitted <= cutoff)]
        if not early.empty:
            continue
        tasks = []
        for assessment in upcoming.to_dict('records'):
            row = snapshot(click_groups.get(key, empty), history, cutoff, assessment)
            tasks.append(dict(name=str(assessment['id_assessment']), features=[row[f] for f in FEATURES],
                              deadline_week=int((assessment['date'] - cutoff) // 7),
                              deadline_day=int(assessment['date'])))
        profiles.append(dict(profile_id=f'profile_{len(profiles)+1:02}', source='OULAD',
                             id_student=int(key[1]), presentation=test_presentation,
                             cutoff_day=cutoff, target=55.0, capacity=3, assignments=tasks))
        if len(profiles) == 30:
            break
    if len(profiles) < 30:
        raise ValueError('Fewer than 30 eligible profiles')
    (out / 'profiles.json').write_text(json.dumps(profiles, indent=2))
    metadata = dict(source='OULAD real data', seed=seed, module=module,
                    train_presentations=presentations[:-1], test_presentation=test_presentation,
                    train_rows=len(train), test_rows=len(test), features=FEATURES,
                    profiles=len(profiles), url=URL, archive_md5=MD5,
                    license='CC BY 4.0', forecast_horizons_days=[14, 28, 42, 70, 112],
                    grade_availability_assumption_days=14,
                    student_overlap=len(set(train.id_student) & set(test.id_student)))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--module', default='BBB')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local-only', action='store_true', help='Use existing CSVs without network access')
    args = parser.parse_args()
    if not args.local_only:
        download(args.raw_dir)
    preprocess(args.raw_dir, args.out_dir, args.module, args.seed)


if __name__ == '__main__':
    main()
