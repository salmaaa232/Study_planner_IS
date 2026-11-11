"""Shared, explicit feature contract. No demographic features are read."""
import numpy as np

GROUPS = ('content', 'communication', 'practice')
CLICK_COLUMNS = [f'clicks_w{week}_{group}' for week in range(1, 5) for group in GROUPS]
FEATURES = CLICK_COLUMNS + ['previous_mean', 'previous_last', 'history_missing',
                           'is_cma', 'weight', 'weeks_since_start', 'horizon_weeks']


def activity_group(activity):
    if activity in {'forumng', 'oucollaborate', 'ouelluminate', 'sharedsubpage'}:
        return 'communication'
    if activity in {'quiz', 'questionnaire', 'ouwiki', 'workshop'}:
        return 'practice'
    return 'content'


def snapshot(clicks, history, cutoff, assessment):
    """Only days strictly before cutoff enter the features.

    OULAD has no grade-release date: previous grades become available under
    a documented assumption of 14 days after max(submission, deadline).
    """
    row = dict.fromkeys(CLICK_COLUMNS, 0.0)
    recent = clicks[(clicks.date < cutoff) & (clicks.date >= cutoff - 28)]
    for item in recent.itertuples():
        week = int((cutoff - 1 - item.date) // 7) + 1
        row[f'clicks_w{week}_{item.group}'] += float(item.sum_click)
    eligible = history[(history.available_day < cutoff) &
                       (history.id_assessment != assessment['id_assessment'])]
    eligible = eligible.sort_values(['available_day', 'id_assessment'])
    row.update(previous_mean=float(eligible.score.mean()) if len(eligible) else 50.0,
               previous_last=float(eligible.score.iloc[-1]) if len(eligible) else 50.0,
               history_missing=int(not len(eligible)),
               is_cma=int(assessment['assessment_type'] == 'CMA'),
               weight=float(assessment['weight']), weeks_since_start=cutoff / 7,
               horizon_weeks=(assessment['date'] - cutoff) / 7)
    return row


def with_sessions(features, sessions, clicks_per_session=40):
    """Generated what-if rule: each session adds 40 content clicks,
    split equally over the four lookback bins. It is NOT a causal effect.
    """
    x = np.asarray(features, dtype=float).copy()
    if sessions < 0 or clicks_per_session <= 0:
        raise ValueError('Sessions must be nonnegative and click increment positive')
    for week in range(4):
        x[week * 3] += sessions * clicks_per_session / 4
    return x
