"""Checks for the supplied real-data/model artifacts, not synthetic fixtures."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from data.features import FEATURES
from nn.model import MarkNetwork

ROOT = Path(__file__).resolve().parents[1]


class RealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(ROOT / 'data/processed/train.csv')
        cls.test = pd.read_csv(ROOT / 'data/processed/test.csv')
        cls.profiles = json.loads((ROOT / 'data/processed/profiles.json').read_text())

    def test_temporal_split_and_disjoint_students(self):
        self.assertEqual(set(self.train.code_presentation), {'2013J'})
        self.assertEqual(set(self.test.code_presentation), {'2014J'})
        self.assertFalse(set(self.train.id_student) & set(self.test.id_student))

    def test_feature_contract(self):
        forbidden = {'gender', 'age_band', 'region', 'disability', 'imd_band', 'final_result', 'score'}
        self.assertFalse(forbidden & set(FEATURES))
        self.assertEqual(len(FEATURES), 19)
        self.assertTrue(np.isfinite(self.train[FEATURES].values).all())
        self.assertTrue(np.isfinite(self.test[FEATURES].values).all())

    def test_profiles_have_covered_horizons(self):
        self.assertEqual(len(self.profiles), 30)
        horizons = set(self.train.horizon_weeks)
        for profile in self.profiles:
            for task in profile['assignments']:
                self.assertIn(task['features'][-1], horizons)
                self.assertEqual(task['features'][-2], profile['cutoff_day']/7)

    def test_saved_model_has_monotone_paths(self):
        m = MarkNetwork.load(ROOT / 'models/model.npz')
        self.assertTrue((m.w1[:12] >= 0).all())
        self.assertTrue((m.w2 >= 0).all())
        self.assertTrue((m.scale > 0).all())

    def test_thirty_output_timetables(self):
        for profile in self.profiles:
            r = json.loads((ROOT / 'outputs/profiles' / f"{profile['profile_id']}.json").read_text())
            slots = [(s['week'], s['slot']) for s in r['timetable']]
            self.assertEqual(len(slots), len(set(slots)))
            self.assertEqual(len(slots), sum(r['allocation']))
            for s in r['timetable']:
                self.assertLessEqual(s['week'], r['deadlines'][s['assignment']-1])
            if r['status'] == 'solution':
                self.assertTrue(all(p >= t for p, t in zip(r['predicted_marks'], r['targets'])))
