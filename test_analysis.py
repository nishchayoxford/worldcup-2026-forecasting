"""Audits for retrospective diagnostics; synthetic data only in unit fixtures."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

import pytest

import model

ROOT = Path(__file__).resolve().parent


def test_paired_cluster_bootstrap_preserves_whole_groups():
    import analysis
    assert hasattr(analysis, 'bootstrap_difference'), 'paired cluster resampling missing'
    # Named synthetic unit fixture: two perfectly dependent pairs.
    values, groups = [-2.0, -2.0, 4.0, 4.0], ['A', 'A', 'B', 'B']
    result = analysis.bootstrap_difference(values, groups, repetitions=4000, seed=19)
    assert result == analysis.bootstrap_difference(values, groups, repetitions=4000, seed=19)
    assert result['difference'] == 1
    assert result['group_cluster']['ci95'] == [-2.0, 4.0]
    assert result['group_cluster']['cluster_count'] == 2
    constant = analysis.bootstrap_difference([0.5] * 4, groups, repetitions=100)
    assert constant['iid']['ci95'] == [0.5, 0.5]
    assert constant['group_cluster']['ci95'] == [0.5, 0.5]
    with pytest.raises(ValueError):
        analysis.bootstrap_difference(values, ['A'], repetitions=100)



def test_wilson_intervals_handle_boundaries_without_fake_precision():
    import analysis
    assert hasattr(analysis, 'wilson'), 'binomial intervals missing'
    assert analysis.wilson(0, 0) is None
    assert analysis.wilson(0, 10) == pytest.approx([0, 0.2775328], abs=1e-6)
    assert analysis.wilson(10, 10) == pytest.approx([0.7224672, 1], abs=1e-6)
    with pytest.raises(ValueError):
        analysis.wilson(3, 2)


def test_build_audits_real_inputs_and_replays_only_frozen_validation():
    import analysis
    assert hasattr(analysis, 'build'), 'diagnostic build missing'
    diagnostic = analysis.build(ROOT, write=False)
    assert diagnostic['counts'] == {'training': 27728, 'validation': 2034, 'refit': 32325, 'test': 72}
    assert diagnostic['audit']['validation_last_date'] < '2024-01-01'
    assert diagnostic['audit']['training_last_date'] < '2022-01-01'
    assert diagnostic['audit']['refit_last_date'] < model.CUTOFF
    assert sum(row['count'] for row in diagnostic['annual_coverage']) == 32325
    assert len(diagnostic['ranking']) == 48
    for name in ('elo', 'poisson'):
        validation = diagnostic['validation'][name]
        assert len(validation['matches']) == 2034
        chosen = min(diagnostic['parameter_grid'][name], key=lambda r: r['log_loss'])
        assert validation['matches'][-1]['cumulative_log_loss'] == pytest.approx(chosen['log_loss'])
        assert len(diagnostic['parameter_grid'][name]) == 12
        assert diagnostic['rating_traces'][name]['ratings'] == pytest.approx(diagnostic['fitted']['ratings'][name])
        assert sum(map(sum, diagnostic['confusion'][name])) == 72
        for bins in diagnostic['calibration'][name].values():
            assert sum(row['count'] for row in bins) == 72
    assert len(diagnostic['comparisons']) == 5
    assert all(not Path(path).is_absolute() for path in diagnostic['input_sha256'])


def test_build_rejects_tampered_nonwinning_validation_candidate(tmp_path):
    import analysis
    for name in ('data/history.csv', 'data/fixtures.csv', 'data/outcomes.csv',
                 'artifacts/predictions.json', 'artifacts/evaluation.json'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    path = tmp_path / 'artifacts/predictions.json'
    prediction = json.loads(path.read_text(encoding='utf-8'))
    prediction['fitted']['tuning']['candidates']['elo'][0]['log_loss'] += 0.05
    path.write_text(json.dumps(prediction), encoding='utf-8')
    evaluation_path = tmp_path / 'artifacts/evaluation.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    evaluation['predictions_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    evaluation_path.write_text(json.dumps(evaluation), encoding='utf-8')
    with pytest.raises(ValueError, match='candidate'):
        analysis.build(tmp_path, write=False)


def test_rating_trace_rejects_postcutoff_rows():
    import analysis
    history, _ = model.load_history(ROOT / 'data/history.csv')
    # Named boundary-mutation fixture: this record must never enter a trace.
    future_fixture = dict(history[-1], date=model.CUTOFF)
    with pytest.raises(ValueError, match='cutoff'):
        analysis.rating_trace([future_fixture], 40, 60)


def test_rating_trace_matches_production_ratings():
    assert importlib.util.find_spec('analysis') is not None, 'rating trace implementation missing'
    from analysis import rating_trace
    history, _ = model.load_history(ROOT / 'data/history.csv')
    trace = rating_trace(history, 40, 60)
    assert trace['ratings'] == pytest.approx(model.elo_ratings(history, 40, 60), abs=1e-10)
    assert all(point['date'] < model.CUTOFF for series in trace['teams'].values() for point in series)
    assert all(series[-1]['rating'] == trace['ratings'][team] for team, series in trace['teams'].items())


def test_full_diagnostics_fit_within_publication_size_budget(tmp_path):
    import analysis
    for name in ('data/history.csv', 'data/fixtures.csv', 'data/outcomes.csv',
                 'artifacts/predictions.json', 'artifacts/evaluation.json'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    result = analysis.build(tmp_path)
    artifact = tmp_path / 'artifacts/diagnostics.json'
    assert artifact.stat().st_size < 10_000_000
    assert json.loads(artifact.read_text(encoding='utf-8')) == result


def test_rating_replay_tolerates_only_negligible_platform_rounding(tmp_path):
    import analysis
    for name in ('data/history.csv', 'data/fixtures.csv', 'data/outcomes.csv',
                 'artifacts/predictions.json', 'artifacts/evaluation.json'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    path = tmp_path / 'artifacts/predictions.json'
    prediction = json.loads(path.read_text(encoding='utf-8'))
    team = next(iter(prediction['fitted']['ratings']['elo']))
    original = prediction['fitted']['ratings']['elo'][team]
    evaluation_path = tmp_path / 'artifacts/evaluation.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    for delta, accepted in [(1e-11, True), (1.0, False)]:
        prediction['fitted']['ratings']['elo'][team] = original + delta
        path.write_text(json.dumps(prediction), encoding='utf-8')
        evaluation['predictions_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        evaluation_path.write_text(json.dumps(evaluation), encoding='utf-8')
        if accepted:
            assert analysis.build(tmp_path, write=False)['audit']['input_hashes_verified']
        else:
            with pytest.raises(ValueError, match='ratings'):
                analysis.build(tmp_path, write=False)
