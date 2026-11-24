"""Integrity checks on the licensed, versioned research inputs."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class DataTests(unittest.TestCase):
    def test_versioned_inputs_are_complete_and_have_a_strict_temporal_cutoff(self):
        self.assertIsNotNone(importlib.util.find_spec('validate_data'), 'data audit module missing')
        import validate_data
        audit = validate_data.validate(ROOT)
        self.assertEqual(audit['history_matches'], 32325)
        self.assertEqual(audit['fixtures'], 72)
        self.assertEqual(audit['teams'], 48)
        self.assertEqual(audit['groups'], 12)
        self.assertEqual(audit['history_last_date'], '2026-06-10')
        self.assertTrue(audit['target_ids_match'])

    def test_out_of_tournament_fixture_is_rejected_even_with_a_new_file_hash(self):
        import validate_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / 'data', root / 'data')
            path = root / 'data/fixtures.csv'
            content = path.read_text(encoding='utf-8').replace('2026-06-11', '2027-06-11', 1)
            path.write_text(content, encoding='utf-8')
            manifest_path = root / 'data/manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['files']['fixtures.csv']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'group-stage date'):
                validate_data.validate(root)


if __name__ == '__main__':
    unittest.main()
