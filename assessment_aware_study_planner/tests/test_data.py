import unittest
import pandas as pd
from data.features import snapshot, FEATURES


class DataTests(unittest.TestCase):
    def test_cutoff_prevents_future_clicks_and_scores(self):
        clicks = pd.DataFrame(dict(date=[-19, 9, 10, 11], group=['content']*4, sum_click=[999, 3, 1000, 2000]))
        history = pd.DataFrame(dict(available_day=[9, 10, 11], id_assessment=[1, 2, 3], score=[60, 0, 100]))
        assessment = dict(id_assessment=4, assessment_type='TMA', weight=20, date=24)
        result = snapshot(clicks, history, 10, assessment)
        self.assertEqual(result['clicks_w1_content'], 3)
        self.assertEqual(result['previous_mean'], 60)
        self.assertEqual(sum(result[k] for k in FEATURES[:12]), 3)

    def test_missing_history(self):
        clicks = pd.DataFrame(columns=['date', 'group', 'sum_click'])
        history = pd.DataFrame(columns=['available_day', 'id_assessment', 'score'])
        a = dict(id_assessment=1, assessment_type='CMA', weight=10, date=14)
        r = snapshot(clicks, history, 0, a)
        self.assertEqual(r['history_missing'], 1)
        self.assertEqual(r['previous_mean'], 50)
        self.assertEqual(r['is_cma'], 1)
