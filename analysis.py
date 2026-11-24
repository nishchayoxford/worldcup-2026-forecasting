"""Reproducible diagnostics for the retrospective World Cup backtest.

Diagnostic code never selects parameters from tournament outcomes. The validation
path reuses training-end ratings without updates, exactly as model.fit_models.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np

import model


def bootstrap_difference(differences, groups, repetitions=4000, seed=2026):
    """Paired mean differences, IID and whole-group percentile intervals.

    Resample G group IDs with replacement and include every match in each selected
    group. Unequal group sizes use a ratio of resampled sums to resampled counts.
    This is not a team/network bootstrap or a population-level guarantee.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) != len(groups) or not len(values):
        raise ValueError('nonempty aligned differences and groups required')
    if not np.isfinite(values).all() or repetitions < 100:
        raise ValueError('finite differences and at least 100 repetitions required')
    rng = np.random.default_rng(seed)
    iid = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    names = sorted(set(groups))
    sums = np.array([sum(v for v, g in zip(values, groups) if g == name) for name in names])
    counts = np.array([groups.count(name) for name in names])
    indices = rng.integers(0, len(names), (repetitions, len(names)))
    cluster = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {'difference': float(values.mean()), 'repetitions': repetitions, 'seed': seed,
            'iid': {'ci95': np.quantile(iid, [0.025, 0.975]).tolist()},
            'group_cluster': {'ci95': np.quantile(cluster, [0.025, 0.975]).tolist(),
                              'cluster_count': len(names)}}


def wilson(successes, count, z=1.959963984540054):
    """Unadjusted binomial 95% score interval; empty bins have no interval."""
    if count < 0 or not 0 <= successes <= count:
        raise ValueError('require 0 <= successes <= count')
    if not count:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * (p * (1 - p) / count + z * z / (4 * count * count)) ** 0.5 / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def rating_trace(history, k, home_advantage):
    """All post-match-day team ratings, preserving simultaneous-day updates."""
    if not history or any(not model.TRAIN_START <= row['date'] < model.CUTOFF for row in history):
        raise ValueError('rating traces require selected history strictly before cutoff')
    ratings, teams = {}, {}
    ordered = sorted(history, key=lambda r: (r['date'], r['home_team'], r['away_team']))
    for day, rows in groupby(ordered, key=lambda r: r['date']):
        changes = {}
        for row in rows:
            home, away = row['home_team'], row['away_team']
            rh, ra = ratings.setdefault(home, 1500.0), ratings.setdefault(away, 1500.0)
            difference = rh - ra + (0 if row['neutral'] else home_advantage)
            expected = 1 / (1 + 10 ** (max(-2400, min(2400, -difference)) / 400))
            delta = k * ([1, 0.5, 0][model.label(row)] - expected)
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, change in changes.items():
            ratings[team] += change
            teams.setdefault(team, []).append({'date': day, 'rating': ratings[team]})
    return {'ratings': dict(sorted(ratings.items())), 'teams': dict(sorted(teams.items()))}


def build(root, write=True):
    """Audit existing forecast/evaluation artifacts and write derived diagnostics."""
    root = Path(root)
    inputs = ['data/history.csv', 'data/fixtures.csv', 'data/outcomes.csv',
              'artifacts/predictions.json', 'artifacts/evaluation.json']
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in inputs}
    prediction = json.loads((root / inputs[3]).read_text(encoding='utf-8'))
    evaluation = json.loads((root / inputs[4]).read_text(encoding='utf-8'))
    history, history_audit = model.load_history(root / inputs[0])
    fixtures = model.load_fixtures(root / inputs[1])
    if (prediction['history']['selected_sha256'] != history_audit['selected_sha256']
            or prediction['fixtures_sha256'] != hashes[inputs[1]]
            or prediction['forecast_sha256'] != model.digest(prediction['forecasts'])
            or evaluation['predictions_sha256'] != hashes[inputs[3]]
            or evaluation['outcomes_sha256'] != hashes[inputs[2]]
            or evaluation['forecast_sha256'] != prediction['forecast_sha256']):
        raise ValueError('input integrity mismatch; regenerate predictions and evaluation')
    matches = evaluation['matches']
    if len(matches) != len(fixtures) or len({r['match_id'] for r in matches}) != len(matches):
        raise ValueError('evaluation must contain each fixture exactly once')
    forecast_by_id = {r['match_id']: r for r in prediction['forecasts']}
    outcomes = model.read_csv(root / inputs[2], model.OUTCOME_FIELDS)
    outcomes_by_id = {r['match_id']: r for r in outcomes}
    if len(outcomes_by_id) != len(outcomes) or set(outcomes_by_id) != set(forecast_by_id):
        raise ValueError('outcome IDs must match forecasts exactly')
    fixture_by_id = {r['match_id']: r for r in fixtures}
    for row in matches:
        forecast = forecast_by_id[row['match_id']]
        if any(row[key] != value for key, value in forecast.items()):
            raise ValueError('evaluation forecast mismatch')
        if any(forecast[key] != value for key, value in fixture_by_id[row['match_id']].items()):
            raise ValueError('fixture metadata mismatch')
        outcome = outcomes_by_id[row['match_id']]
        if any(row[key] != model.score(outcome[key]) for key in ('home_score', 'away_score')):
            raise ValueError('evaluation outcome mismatch')
        if row['actual_class'] != model.CLASSES[model.label(row)]:
            raise ValueError('evaluation class mismatch')
    fitted = prediction['fitted']
    train = [r for r in history if r['date'] < '2022-01-01']
    valid = [r for r in history if '2022-01-01' <= r['date'] < '2024-01-01']
    # Replay the complete declared grid, including losing candidates. fit_models
    # validates its date boundary and never sees tournament outcomes.
    replay = model.fit_models(history)
    for name in ('elo', 'poisson'):
        stored = fitted['tuning']['candidates'][name]
        expected = replay['tuning']['candidates'][name]
        if len(stored) != len(expected) or any(
                a['parameters'] != b['parameters']
                or not math.isclose(a['log_loss'], b['log_loss'], rel_tol=0, abs_tol=1e-12)
                for a, b in zip(stored, expected)):
            raise ValueError('validation candidate grid does not match pre-cutoff replay')
    trace_cache, validation, traces = {}, {}, {}
    for name in ('elo', 'poisson'):
        params = fitted['parameters'][name]
        cache_key = (params['k'], params['home_advantage'])
        if cache_key not in trace_cache:
            trace_cache[cache_key] = rating_trace(history, *cache_key)
        traces[name] = trace_cache[cache_key]
        reproduced_ratings = traces[name]['ratings']
        stored_ratings = fitted['ratings'][name]
        if set(reproduced_ratings) != set(stored_ratings) or any(
                not math.isclose(value, stored_ratings[team], rel_tol=0, abs_tol=1e-9)
                for team, value in reproduced_ratings.items()):
            raise ValueError('stored cutoff ratings do not match historical replay')
        ratings = model.elo_ratings(train, *cache_key)
        mean = model.average_goals(train)
        rows, total = [], 0.0
        for row in valid:
            delta = ratings.get(row['home_team'], 1500) - ratings.get(row['away_team'], 1500)
            delta += 0 if row['neutral'] else params['home_advantage']
            p = (model.elo_probabilities(delta, params['draw_rate']) if name == 'elo'
                 else model.poisson_probabilities(delta, mean, params['scale']))
            loss = -math.log(p[model.label(row)])
            total += loss
            rows.append({'date': row['date'], 'home_team': row['home_team'], 'away_team': row['away_team'],
                         'probabilities': p, 'actual_class': model.CLASSES[model.label(row)],
                         'log_loss': loss, 'cumulative_log_loss': total / (len(rows) + 1)})
        trials = fitted['tuning']['candidates'][name]
        selected = min(trials, key=lambda r: r['log_loss'])
        if selected['parameters'] != params or not math.isclose(total / len(rows), selected['log_loss'], abs_tol=1e-12):
            raise ValueError('selected parameters or validation scores mismatch')
        validation[name] = {'protocol': 'Frozen 2021-end ratings; chronological scoring, not model updates or epochs.',
                            'training_mean_goals': mean, 'matches': rows}
    teams = sorted({r[key] for r in fixtures for key in ('home_team', 'away_team')})
    ranking = sorted([{'team': team, 'rating': fitted['ratings']['elo'][team]} for team in teams],
                     key=lambda r: (-r['rating'], r['team']))
    years = Counter(r['date'][:4] for r in history)
    y = [model.CLASSES.index(r['actual_class']) for r in matches]
    metrics, losses, calibration, confusion = {}, {}, {}, {}
    for name in ('elo', 'poisson', 'uniform', 'historical'):
        probabilities = [r['probabilities'][name] for r in matches]
        metrics[name] = model.metrics(probabilities, y)
        for metric in ('log_loss', 'brier_sum', 'accuracy'):
            if not math.isclose(metrics[name][metric], evaluation['models'][name][metric], abs_tol=1e-12):
                raise ValueError('evaluation metric mismatch')
        losses[name] = model.per_match_metrics(probabilities, y)
        if name not in ('elo', 'poisson'):
            continue
        cal = metrics[name]['calibration']
        calibration[name] = {'top_label': cal['top_label_bins'],
                             **{key: value['bins'] for key, value in cal['by_class'].items()}}
        for bins in calibration[name].values():
            for item in bins:
                successes = round(item['observed_frequency'] * item['count']) if item['count'] else 0
                item['wilson_ci95'] = wilson(successes, item['count'])
        matrix = [[0] * 3 for _ in range(3)]
        for p, actual in zip(probabilities, y):
            matrix[actual][max(range(3), key=lambda k: p[k])] += 1
        confusion[name] = matrix
    groups = [r['group'] for r in matches]
    comparisons = {}
    pairs = [(n, b) for n in ('elo', 'poisson') for b in ('uniform', 'historical')] + [('poisson', 'elo')]
    for name, baseline in pairs:
        differences = [a - b for a, b in zip(losses[name]['log_loss'], losses[baseline]['log_loss'])]
        comparisons[f'{name}_minus_{baseline}'] = bootstrap_difference(differences, groups)
    group_scores = []
    for group in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == group]
        group_scores.append({'group': group, 'count': len(indices),
                             **{name: float(np.mean([losses[name]['log_loss'][i] for i in indices]))
                                for name in losses}})
    result = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'study_type': prediction['study_type'], 'input_sha256': hashes,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(model.__file__))},
        'counts': {'training': len(train), 'validation': len(valid), 'refit': len(history), 'test': len(matches)},
        'audit': {'training_last_date': train[-1]['date'], 'validation_last_date': valid[-1]['date'],
                  'refit_last_date': history[-1]['date'], 'cutoff_exclusive': model.CUTOFF,
                  'test_first_date': min(r['date'] for r in matches), 'test_last_date': max(r['date'] for r in matches),
                  'history': history_audit, 'validation_ratings_frozen': True,
                  'test_outcomes_used_for_tuning': False, 'input_hashes_verified': True},
        'annual_coverage': [{'year': int(year), 'count': count} for year, count in sorted(years.items())],
        'fitted': fitted, 'rating_traces': traces, 'ranking': ranking,
        'trajectory_selection': [r['team'] for r in ranking[:8]],
        'parameter_grid': fitted['tuning']['candidates'], 'validation': validation,
        'metrics': metrics, 'calibration': calibration, 'confusion': confusion,
        'matches': matches, 'group_scores': group_scores, 'comparisons': comparisons,
        'uncertainty': {
            'calibration': '95% Wilson binomial intervals, unadjusted for repeated-team/group dependence; descriptive only.',
            'bootstrap': '4000 paired resamples, NumPy default_rng seed 2026, linear percentile 95% intervals; model minus reference log loss.',
            'cluster': 'Sample G group IDs with replacement; retain all matches per sampled group; ratio of sums/counts. Only 12 groups; not team/network robust.',
            'scope': 'Conditional on fitted models. No refitting or parameter-selection uncertainty. Not betting edge or profit.'},
        'limitations': evaluation['limitations'],
    }
    if write:
        destination = root / 'artifacts/diagnostics.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Compact serialization preserves every trace without bloating the public repository.
        destination.write_text(json.dumps(result, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = build(args.root)
    print(f"Wrote artifacts/diagnostics.json; counts={output['counts']}")
