"""End-to-end checks use the versioned real dataset, not simulated performance."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ('predictions.json', 'evaluation.json', 'summary.json', 'match_predictions.csv')


class PipelineTests(unittest.TestCase):
    def assert_json_mutation_rejected(self, artifact, keys, value):
        import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in ARTIFACTS:
                shutil.copy2(ROOT / 'artifacts' / name, output / name)
            path = output / artifact
            payload = json.loads(path.read_text(encoding='utf-8'))
            target = payload
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            path.write_text(json.dumps(payload), encoding='utf-8')
            for left, right in ((ROOT / 'artifacts', output), (output, ROOT / 'artifacts')):
                with self.assertRaises(ValueError):
                    run_pipeline.compare(left, right)

    def test_evaluation_deterministic_contents_are_compared(self):
        mutations = [
            (('models', 'elo', 'calibration', 'top_label_ece'), 0.9),
            (('bootstrap', 'comparisons', 'elo_minus_uniform', 'log_loss', 'ci95', 0), -0.99),
            (('matches', 0, 'actual_class'), 'away_win'),
        ]
        for keys, value in mutations:
            with self.subTest(keys=keys):
                self.assert_json_mutation_rejected('evaluation.json', keys, value)

    def test_prediction_metadata_and_fitted_outputs_are_compared(self):
        mutations = [
            (('fixtures_sha256',), '0' * 64),
            (('history', 'raw_sha256'), '0' * 64),
            (('fitted', 'ratings', 'elo', 'Mexico'), 1000.0),
            (('audit', 'outcomes_read'), True),
        ]
        for keys, value in mutations:
            with self.subTest(keys=keys):
                self.assert_json_mutation_rejected('predictions.json', keys, value)

    def test_forecast_hashes_are_validated_not_blindly_excluded(self):
        for artifact in ARTIFACTS[:3]:
            with self.subTest(artifact=artifact):
                self.assert_json_mutation_rejected(artifact, ('forecast_sha256',), '0' * 64)

    def test_invalid_evaluation_probabilities_fail_even_when_identical(self):
        import run_pipeline
        for probabilities in ([0.5, 0.5], [0.5, 0.5, 0.5], [float('nan'), 0.5, 0.5]):
            with self.subTest(probabilities=probabilities), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                for name in ARTIFACTS:
                    shutil.copy2(ROOT / 'artifacts' / name, output / name)
                path = output / 'evaluation.json'
                payload = json.loads(path.read_text(encoding='utf-8'))
                payload['matches'][0]['probabilities']['elo'] = probabilities
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    run_pipeline.compare(output, output)

    def test_source_hashes_schemas_and_ids_remain_exact(self):
        mutations = [
            ('predictions.json', ('history', 'selected_sha256'), '0' * 64),
            ('evaluation.json', ('outcomes_sha256',), '0' * 64),
            ('summary.json', ('data_audit', 'input_hashes', 'history.csv'), '0' * 64),
            ('evaluation.json', ('matches', 0, 'match_id'), '0760415'),
            ('evaluation.json', ('matches',), []),
            ('evaluation.json', ('models', 'elo', 'calibration'), {}),
            ('summary.json', ('evaluated_matches',), 72.0),
            ('summary.json', ('data_audit', 'target_ids_match'), 1),
            ('summary.json', ('data_audit', 'created_utc'), 'nested timestamp is not excluded'),
            ('summary.json', ('created_utc',), []),
        ]
        for artifact, keys, value in mutations:
            with self.subTest(artifact=artifact, keys=keys):
                self.assert_json_mutation_rejected(artifact, keys, value)

    def test_roundoff_and_root_execution_metadata_are_allowed(self):
        import model
        import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            payloads = {name: json.loads((ROOT / 'artifacts' / name).read_text(encoding='utf-8'))
                        for name in ARTIFACTS[:3]}
            prediction = payloads['predictions.json']
            p = prediction['forecasts'][0]['probabilities']['elo']
            p[0] += 1e-16
            p[1] -= 1e-16
            forecast_hash = model.digest(prediction['forecasts'])
            self.assertNotEqual(forecast_hash, prediction['forecast_sha256'])
            payloads['evaluation.json']['matches'][0]['probabilities']['elo'] = p
            payloads['evaluation.json']['models']['elo']['calibration']['top_label_ece'] += 1e-13
            payloads['evaluation.json']['bootstrap']['comparisons']['elo_minus_uniform']['log_loss']['ci95'][0] += 1e-13
            for name, payload in payloads.items():
                payload['created_utc'] = '2026-11-25T00:00:00+00:00'
                payload['forecast_sha256'] = forecast_hash
                if 'code_sha256' in payload:
                    payload['code_sha256'] = '1' * 64
                if 'predictions_sha256' in payload:
                    payload['predictions_sha256'] = '2' * 64
                run_pipeline.save_json(output / name, payload)
            with (ROOT / 'artifacts/match_predictions.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            rows[0]['elo_p_home_win'], rows[0]['elo_p_draw'] = p[:2]
            with (output / 'match_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertTrue(run_pipeline.compare(ROOT / 'artifacts', output))
            self.assertTrue(run_pipeline.compare(output, ROOT / 'artifacts'))

    def test_all_four_artifacts_are_required(self):
        import run_pipeline
        for missing in ARTIFACTS:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                for name in ARTIFACTS:
                    if name != missing:
                        shutil.copy2(ROOT / 'artifacts' / name, output / name)
                with self.assertRaises(FileNotFoundError):
                    run_pipeline.compare(ROOT / 'artifacts', output)

    def test_summary_correct_count_is_compared(self):
        self.assert_json_mutation_rejected('summary.json', ('models', 'elo', 'correct'), 45)

    def test_exported_csv_probabilities_and_top_picks_are_compared(self):
        import run_pipeline
        for mutation in ('probabilities', 'top_pick'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                for name in ARTIFACTS:
                    shutil.copy2(ROOT / 'artifacts' / name, output / name)
                path = output / 'match_predictions.csv'
                with path.open(encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                if mutation == 'probabilities':
                    rows[0]['elo_p_home_win'] = float(rows[0]['elo_p_home_win']) - 0.01
                    rows[0]['elo_p_draw'] = float(rows[0]['elo_p_draw']) + 0.01
                else:
                    rows[0]['elo_top_pick'] = 'away_win'
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                for left, right in ((ROOT / 'artifacts', output), (output, ROOT / 'artifacts')):
                    with self.assertRaises(ValueError):
                        run_pipeline.compare(left, right)

    def test_reproduce_writes_complete_results_and_matches_the_reference(self):
        self.assertIsNotNone(importlib.util.find_spec('run_pipeline'), 'pipeline module missing')
        import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run_pipeline.build(ROOT, output)
            self.assertEqual(summary['evaluated_matches'], 72)
            self.assertEqual(summary['models']['elo']['correct'], 44)
            self.assertEqual(summary['models']['poisson']['correct'], 44)
            self.assertTrue((output / 'match_predictions.csv').is_file())
            prediction = json.loads((output / 'predictions.json').read_text(encoding='utf-8'))
            self.assertFalse(prediction['audit']['outcomes_read'])
            self.assertTrue(run_pipeline.compare(ROOT / 'artifacts', output))

    def test_reference_probability_shape_is_validated_not_just_its_hash(self):
        import model
        import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in ARTIFACTS:
                shutil.copy2(ROOT / 'artifacts' / name, output / name)
            path = output / 'predictions.json'
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['forecasts'][0]['probabilities']['elo'].pop()
            payload['forecast_sha256'] = model.digest(payload['forecasts'])
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                run_pipeline.compare(output, ROOT / 'artifacts')


if __name__ == '__main__':
    unittest.main()
