"""End-to-end checks for the publication figure bundle on real inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parent
STEMS = [
    '01_research_design', '02_data_coverage', '03_rating_trajectories',
    '04_team_ranking', '05_parameter_search', '06_validation_path',
    '07_model_comparison', '08_calibration', '09_confusion_matrices',
    '10_match_probabilities', '11_group_performance', '12_uncertainty',
    '13_forecast_errors',
]


def test_build_renders_complete_real_data_bundle(tmp_path):
    assert importlib.util.find_spec('plots') is not None, 'publication renderer missing'
    import plots
    for name in ('data/history.csv', 'data/fixtures.csv', 'data/outcomes.csv',
                 'artifacts/predictions.json', 'artifacts/evaluation.json'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    manifest = plots.build(tmp_path)
    assert [entry['id'] for entry in manifest['figures']] == STEMS
    assert manifest == json.loads((tmp_path / 'artifacts/figure_manifest.json').read_text(encoding='utf-8'))
    assert manifest['test_match_count'] == 72
    for entry in manifest['figures']:
        assert entry['caption'] and entry['title'] and entry['data_keys']
        for extension in ('png', 'svg'):
            path = Path(entry[extension])
            assert not path.is_absolute()
            payload = (tmp_path / path).read_bytes()
            assert len(payload) > 10000
            assert hashlib.sha256(payload).hexdigest() == entry['sha256'][extension]
        with Image.open(tmp_path / entry['png']) as image:
            assert image.width >= 1500 and image.height >= 700
            assert image.getbbox() is not None
        text = (tmp_path / entry['svg']).read_text(encoding='utf-8')
        ET.fromstring(text)
        assert '<dc:date>' not in text
        assert 'C:\\Users' not in text and 'C:/Users' not in text
