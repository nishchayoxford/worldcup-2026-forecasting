"""All generated match records in this suite are SYNTHETIC TEST FIXTURES, not results."""
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
HISTORY_FIELDS = 'date,home_team,away_team,home_score,away_score,tournament,city,country,neutral'.split(',')
FIXTURE_FIELDS = 'match_id,date,group,home_team,away_team,home_advantage'.split(',')


def synthetic_history():
    rows = []
    for year in range(2018, 2025):
        for month, home, away, hs, aws, neutral in [
            (1, 'Alpha', 'Beta', 2, 0, 'FALSE'),
            (3, 'Beta', 'Gamma', 1, 1, 'TRUE'),
            (5, 'Gamma', 'Alpha', 0, 1, 'TRUE'),
            (7, 'Beta', 'Alpha', 1, 2, 'FALSE'),
            (9, 'Alpha', 'Gamma', 0, 0, 'TRUE'),
            (11, 'Gamma', 'Beta', 2, 1, 'FALSE'),
        ]:
            rows.append(dict(zip(HISTORY_FIELDS, [f'{year}-{month:02d}-01', home, away, hs, aws, 'Synthetic test only', 'Test', 'Test', neutral])))
    return rows


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.history = self.root / 'history.csv'
        write_csv(self.history, HISTORY_FIELDS, synthetic_history())

    def model(self):
        self.assertTrue((ROOT / 'model.py').exists(), 'history loader implementation is missing')
        return importlib.import_module('model')

    def test_history_explicitly_excludes_future_and_hashes_selected_rows(self):
        m = self.model()
        rows, audit = m.load_history(self.history)
        future = synthetic_history()[0] | {'date': '2026-06-11', 'home_score': 50}
        write_csv(self.history, HISTORY_FIELDS, synthetic_history() + [future])
        selected, updated = m.load_history(self.history)
        self.assertEqual(rows, selected)
        self.assertEqual(audit['selected_sha256'], updated['selected_sha256'])
        self.assertNotEqual(audit['raw_sha256'], updated['raw_sha256'])
        self.assertEqual(updated['excluded_on_or_after_cutoff'], 1)
        self.assertTrue(all(row['date'] < m.CUTOFF for row in selected))


    def test_history_fail_closed_for_duplicates_and_malformed_values(self):
        m = self.model()
        good = synthetic_history()
        for bad in [good + [good[0]], [good[0] | {'neutral': 'maybe'}],
                    [good[0] | {'home_score': '-1'}], [good[0] | {'home_team': ''}],
                    [good[0] | {'date': '2022-1-1'}], [good[0] | {'home_score': '1.5'}],
                    [good[0] | {'away_team': 'Alpha'}]]:
            with self.subTest(bad=bad[-1]):
                write_csv(self.history, HISTORY_FIELDS, bad)
                with self.assertRaises(ValueError):
                    m.load_history(self.history)
        write_csv(self.history, HISTORY_FIELDS[:-1], [{k: v for k, v in good[0].items() if k != 'neutral'}])
        with self.assertRaises(ValueError):
            m.load_history(self.history)


    def test_probability_models_are_symmetric_normalized_and_strength_sensitive(self):
        m = self.model()
        self.assertTrue(hasattr(m, 'elo_probabilities'), 'probability models are missing')
        for predict in [lambda d: m.elo_probabilities(d, 0.28),
                        lambda d: m.poisson_probabilities(d, 1.35, 1.0)]:
            for diff in [0, 100, 800, 10000]:
                p, q = predict(diff), predict(-diff)
                self.assertAlmostEqual(sum(p), 1, places=12)
                self.assertTrue(all(math.isfinite(v) and 0 < v < 1 for v in p))
                for a, b in zip(p, reversed(q)):
                    self.assertAlmostEqual(a, b, places=12)
                if diff > 0:
                    self.assertGreater(p[0], p[2])
            self.assertGreater(predict(75)[0], predict(0)[0])
        self.assertAlmostEqual(m.elo_probabilities(0, 0.28)[1], 0.28)
        equal_poisson = m.poisson_probabilities(0, 1.0, 1.0)
        self.assertAlmostEqual(equal_poisson[1], 0.308508322553671, places=10)


    def test_metrics_calibration_and_paired_bootstrap_have_known_values(self):
        m = self.model()
        self.assertTrue(hasattr(m, 'metrics'), 'evaluation metrics are missing')
        uniform = [[1 / 3] * 3] * 3
        labels = [0, 1, 2]
        result = m.metrics(uniform, labels)
        self.assertAlmostEqual(result['log_loss'], math.log(3))
        self.assertAlmostEqual(result['brier_sum'], 2 / 3)
        self.assertAlmostEqual(result['accuracy'], 1 / 3)
        self.assertEqual(sum(b['count'] for b in result['calibration']['top_label_bins']), 3)
        self.assertAlmostEqual(result['calibration']['top_label_ece'], 0)
        ci = m.paired_bootstrap(uniform, uniform, labels, repetitions=100, seed=7)
        self.assertEqual(ci, m.paired_bootstrap(uniform, uniform, labels, repetitions=100, seed=7))
        for name in ['log_loss', 'brier_sum', 'accuracy']:
            self.assertEqual(ci[name], {'difference': 0.0, 'ci95': [0.0, 0.0]})
        good = [[0.8, 0.1, 0.1]] * 3
        constant = m.paired_bootstrap(good, uniform, [0, 0, 0], repetitions=100, seed=7)
        self.assertAlmostEqual(constant['log_loss']['difference'], -math.log(0.8) - math.log(3))
        self.assertEqual(constant['log_loss']['ci95'][0], constant['log_loss']['ci95'][1])
        for bad_probs, bad_y in [([], []), ([[0.2, 0.2, 0.2]], [0]),
                                 ([[float('nan'), 0, 1]], [0]), (uniform, [0]),
                                 ([[0.3, 0.3, 0.4]], [3])]:
            with self.assertRaises(ValueError):
                m.metrics(bad_probs, bad_y)


    def test_chronological_tuning_freezes_validation_then_refits_all_history(self):
        m = self.model()
        self.assertTrue(hasattr(m, 'fit_models'), 'chronological model fitting is missing')
        history, _ = m.load_history(self.history)
        fitted = m.fit_models(history)
        changed = [r | {'home_score': 0, 'away_score': 9} if r['date'] >= '2024-01-01' else r for r in history]
        updated = m.fit_models(changed)
        self.assertEqual(fitted['parameters'], updated['parameters'])
        self.assertEqual(fitted['tuning'], updated['tuning'])
        self.assertNotEqual(fitted['ratings'], updated['ratings'])
        self.assertEqual(fitted, m.fit_models(list(reversed(history))))
        self.assertEqual(fitted['tuning']['validation_start'], '2022-01-01')
        self.assertEqual(fitted['tuning']['validation_end_exclusive'], '2024-01-01')
        self.assertEqual(fitted['tuning']['validation_count'], 12)
        self.assertLess(fitted['tuning']['training_last_date'], '2022-01-01')
        self.assertEqual(fitted['fit_count'], len(history))
        for name in ['elo', 'poisson']:
            trials = fitted['tuning']['candidates'][name]
            chosen = fitted['parameters'][name]
            best = min(trials, key=lambda r: r['log_loss'])
            self.assertEqual(chosen, best['parameters'])
        p = m.forecast(fitted, 'Alpha', 'Beta', 1)
        q = m.forecast(fitted, 'Beta', 'Alpha', -1)
        neutral = m.forecast(fitted, 'Alpha', 'Beta', 0)
        for name in ['elo', 'poisson', 'uniform', 'historical']:
            self.assertAlmostEqual(sum(p[name]), 1)
            for a, b in zip(p[name], reversed(q[name])):
                self.assertAlmostEqual(a, b)
        self.assertGreaterEqual(p['elo'][0], neutral['elo'][0])
        with self.assertRaises(ValueError):
            m.fit_models(history + [history[0] | {'date': m.CUTOFF}])
        with self.assertRaises(ValueError):
            m.fit_models([r for r in history if r['date'] < '2022-01-01'])
        with self.assertRaises(ValueError):
            m.forecast(fitted, 'Unknown Team', 'Beta', 0)
        single = m.elo_ratings([history[0]], k=20, home_advantage=0)
        self.assertAlmostEqual(single['Alpha'], 1510)
        self.assertAlmostEqual(single['Beta'], 1490)


    def synthetic_fixtures(self):
        return [dict(zip(FIXTURE_FIELDS, values)) for values in [
            ('test-1', '2026-06-11', 'A', 'Alpha', 'Beta', 0),
            ('test-2', '2026-06-18', 'A', 'Alpha', 'Beta', 0),
            ('test-3', '2026-06-20', 'A', 'Gamma', 'Alpha', -1),
        ]]

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'model.py'), *map(str, args)],
                              cwd=self.root, capture_output=True, text=True)

    def test_predict_cli_is_frozen_outcome_independent_and_future_mutation_invariant(self):
        fixture_path, output = self.root / 'fixtures.csv', self.root / 'predictions.json'
        write_csv(fixture_path, FIXTURE_FIELDS, self.synthetic_fixtures())
        command = ('predict', '--history', self.history, '--fixtures', fixture_path, '--output', output)
        before = datetime.now(timezone.utc)
        completed = self.run_cli(*command)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output.exists(), 'predict CLI did not write an artifact')
        initial = json.loads(output.read_text())
        created = datetime.fromisoformat(initial['created_utc'])
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, datetime.now(timezone.utc))
        self.assertEqual(initial['cutoff_exclusive'], '2026-06-11')
        self.assertTrue(initial['audit']['ratings_frozen'])
        self.assertFalse(initial['audit']['outcomes_read'])
        self.assertEqual(initial['forecasts'][0]['probabilities'], initial['forecasts'][1]['probabilities'])
        self.assertEqual(initial['forecast_sha256'], self.model().digest(initial['forecasts']))
        self.assertNotIn('home_score', initial['forecasts'][0])
        (self.root / 'outcomes.csv').write_text('INVALID OUTCOMES MUST NEVER BE READ')
        future = synthetic_history()[0] | {'date': '2026-06-11', 'home_score': 99}
        write_csv(self.history, HISTORY_FIELDS, synthetic_history() + [future])
        self.assertEqual(self.run_cli(*command).returncode, 0)
        updated = json.loads(output.read_text())
        self.assertEqual(initial['forecasts'], updated['forecasts'])
        self.assertEqual(initial['fitted'], updated['fitted'])
        self.assertEqual(initial['history']['selected_sha256'], updated['history']['selected_sha256'])
        self.assertEqual(updated['history']['excluded_on_or_after_cutoff'], 1)
        self.assertEqual(len(initial['code_sha256']), 64)

    def test_evaluate_cli_scores_sealed_forecasts_and_rejects_bad_joins(self):
        fixtures = self.root / 'fixtures.csv'
        predictions = self.root / 'predictions.json'
        outcomes = self.root / 'outcomes.csv'
        report = self.root / 'evaluation.json'
        write_csv(fixtures, FIXTURE_FIELDS, self.synthetic_fixtures())
        self.assertEqual(self.run_cli('predict', '--history', self.history, '--fixtures', fixtures,
                                      '--output', predictions).returncode, 0)
        rows = [{'match_id': f'test-{i}', 'home_score': i, 'away_score': 1} for i in range(1, 4)]
        write_csv(outcomes, ['match_id', 'home_score', 'away_score'], rows)
        command = ('evaluate', '--predictions', predictions, '--outcomes', outcomes, '--output', report)
        done = self.run_cli(*command)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(report.read_text())
        self.assertEqual(result['count'], 3)
        self.assertEqual(set(result['models']), {'elo', 'poisson', 'historical', 'uniform'})
        self.assertAlmostEqual(result['models']['uniform']['log_loss'], math.log(3))
        self.assertEqual(len(result['matches']), 3)
        for bad in [rows[:-1], rows + [rows[0]], rows + [rows[0] | {'match_id': 'extra'}]]:
            write_csv(outcomes, ['match_id', 'home_score', 'away_score'], bad)
            self.assertNotEqual(self.run_cli(*command).returncode, 0)
        write_csv(outcomes, ['match_id', 'home_score', 'away_score'], rows)
        changed = json.loads(predictions.read_text())
        changed['forecasts'][0]['home_team'] = 'Tampered'
        predictions.write_text(json.dumps(changed))
        self.assertNotEqual(self.run_cli(*command).returncode, 0)


if __name__ == '__main__':
    unittest.main()
