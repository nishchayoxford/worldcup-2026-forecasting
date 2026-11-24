"""Pre-cutoff international football forecasts. Standard library only."""
import csv
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import re
import random
from itertools import groupby
from pathlib import Path

CUTOFF = '2026-06-11'  # exclusive, intentionally not a CLI option
TRAIN_START = '1990-01-01'
HISTORY_FIELDS = 'date,home_team,away_team,home_score,away_score,tournament,city,country,neutral'.split(',')
FIXTURE_FIELDS = 'match_id,date,group,home_team,away_team,home_advantage'.split(',')
OUTCOME_FIELDS = 'match_id,home_score,away_score'.split(',')
CLASSES = ['home_win', 'draw', 'away_win']


def normalize(values):
    clipped = [max(1e-15, v) for v in values]
    total = sum(clipped)
    return [v / total for v in clipped]


def elo_probabilities(difference, draw_rate):
    """Davidson draw extension: draw_rate is P(draw) for equal strengths."""
    strength = 10 ** (max(-2400, min(2400, difference)) / 800)
    nu = 2 * draw_rate / (1 - draw_rate)
    return normalize([strength, nu, 1 / strength])


def poisson_probabilities(difference, mean_goals, scale):
    """Independent goals, constant total intensity, Elo-driven goal share."""
    ratio = math.exp(max(-20, min(20, scale * difference / 400)))
    home_mean = 2 * mean_goals * ratio / (1 + ratio)
    away_mean = 2 * mean_goals / (1 + ratio)

    def mass(mu):
        values = [math.exp(-mu)]
        while sum(values) < 1 - 1e-13:
            values.append(values[-1] * mu / len(values))
            if len(values) > 500:
                raise ValueError('Poisson intensity outside supported range')
        return values

    home, away = mass(home_mean), mass(away_mean)
    probs = [0.0, 0.0, 0.0]
    for h, ph in enumerate(home):
        for a, pa in enumerate(away):
            probs[0 if h > a else 1 if h == a else 2] += ph * pa
    return normalize(probs)


def per_match_metrics(probabilities, labels):
    if not labels or len(probabilities) != len(labels):
        raise ValueError('nonempty, aligned predictions and labels required')
    result = {'log_loss': [], 'brier_sum': [], 'accuracy': []}
    for p, y in zip(probabilities, labels):
        if y not in (0, 1, 2) or len(p) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in p) or abs(sum(p) - 1) > 1e-9:
            raise ValueError('invalid class or probability vector')
        result['log_loss'].append(-math.log(max(1e-15, p[y])))
        result['brier_sum'].append(sum((v - int(k == y)) ** 2 for k, v in enumerate(p)))
        result['accuracy'].append(float(max(range(3), key=lambda k: p[k]) == y))
    return result


def metrics(probabilities, labels):
    losses = per_match_metrics(probabilities, labels)
    n = len(labels)

    def reliability(confidences, events):
        bins = []
        for index in range(10):
            selected = [i for i, p in enumerate(confidences) if min(9, int(p * 10)) == index]
            count = len(selected)
            bins.append({'lower': index / 10, 'upper': (index + 1) / 10, 'count': count,
                         'mean_probability': sum(confidences[i] for i in selected) / count if count else None,
                         'observed_frequency': sum(events[i] for i in selected) / count if count else None})
        ece = sum(b['count'] / n * abs(b['mean_probability'] - b['observed_frequency']) for b in bins if b['count'])
        return bins, ece

    winners = [max(range(3), key=lambda k: p[k]) for p in probabilities]
    bins, ece = reliability([max(p) for p in probabilities], [int(y == w) for y, w in zip(labels, winners)])
    by_class = {}
    for k, name in enumerate(CLASSES):
        cb, ce = reliability([p[k] for p in probabilities], [int(y == k) for y in labels])
        by_class[name] = {'bins': cb, 'ece': ce}
    return {'count': n, **{name: sum(values) / n for name, values in losses.items()},
            'calibration': {'top_label_bins': bins, 'top_label_ece': ece, 'by_class': by_class}}


def paired_bootstrap(a, b, labels, repetitions=2000, seed=2026):
    """IID paired match resampling; does not account for team/group dependence."""
    if repetitions < 100:
        raise ValueError('at least 100 bootstrap repetitions required')
    la, lb = per_match_metrics(a, labels), per_match_metrics(b, labels)
    n = len(labels)
    differences = {key: [x - y for x, y in zip(la[key], lb[key])] for key in la}
    samples = {key: [] for key in la}
    rng = random.Random(seed)
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        for key, values in differences.items():
            samples[key].append(sum(values[i] for i in indices) / n)

    def quantile(values, q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {key: {'difference': sum(values) / n, 'ci95': [quantile(sorted(samples[key]), 0.025), quantile(sorted(samples[key]), 0.975)]}
            for key, values in differences.items()}


def label(row):
    h, a = row['home_score'], row['away_score']
    return 0 if h > a else 1 if h == a else 2


def elo_ratings(history, k, home_advantage):
    """Constant-K Elo, 1500 initialization; same-day matches update as a batch."""
    ratings = {}
    ordered = sorted(history, key=lambda r: (r['date'], r['home_team'], r['away_team']))
    for _, day_rows in groupby(ordered, key=lambda r: r['date']):
        changes = {}
        for row in day_rows:
            h, a = row['home_team'], row['away_team']
            rh, ra = ratings.setdefault(h, 1500.0), ratings.setdefault(a, 1500.0)
            difference = rh - ra + (0 if row['neutral'] else home_advantage)
            expected = 1 / (1 + 10 ** (max(-2400, min(2400, -difference)) / 400))
            observed = [1, 0.5, 0][label(row)]
            delta = k * (observed - expected)
            changes[h] = changes.get(h, 0.0) + delta
            changes[a] = changes.get(a, 0.0) - delta
        for team, change in changes.items():
            ratings[team] += change
    return dict(sorted(ratings.items()))


def average_goals(history):
    return max(0.05, sum(r['home_score'] + r['away_score'] for r in history) / (2 * len(history)))


def frequency_baseline(history):
    # Add-one smoothing. Neutral matches have arbitrary home/away labels.
    neutral = [1.0, 1.0, 1.0]
    home = [1.0, 1.0, 1.0]
    for row in history:
        y = label(row)
        if row['neutral']:
            if y == 1:
                neutral[1] += 1
            else:
                neutral[0] += 0.5
                neutral[2] += 0.5
        else:
            home[y] += 1
    return {'neutral': normalize(neutral), 'home': normalize(home)}


def fit_models(history):
    if not history or any(not TRAIN_START <= r['date'] < CUTOFF for r in history):
        raise ValueError('fit_models requires selected history strictly before cutoff')
    history = sorted(history, key=lambda r: (r['date'], r['home_team'], r['away_team']))
    train = [r for r in history if r['date'] < '2022-01-01']
    validation = [r for r in history if '2022-01-01' <= r['date'] < '2024-01-01']
    if not train or not validation:
        raise ValueError('need training before 2022 and validation during 2022-2023')
    mean = average_goals(train)
    candidates = {'elo': [], 'poisson': []}
    y = [label(r) for r in validation]
    for k in (20, 40):
        for home in (0, 60):
            ratings = elo_ratings(train, k, home)
            # Previously unseen validation teams start at the predeclared prior.
            differences = [ratings.get(r['home_team'], 1500) - ratings.get(r['away_team'], 1500) + (0 if r['neutral'] else home) for r in validation]
            for draw in (0.22, 0.28, 0.34):
                probabilities = [elo_probabilities(d, draw) for d in differences]
                loss = sum(per_match_metrics(probabilities, y)['log_loss']) / len(y)
                candidates['elo'].append({'parameters': {'k': k, 'home_advantage': home, 'draw_rate': draw}, 'log_loss': loss})
            for scale in (0.75, 1.25, 1.75):
                probabilities = [poisson_probabilities(d, mean, scale) for d in differences]
                loss = sum(per_match_metrics(probabilities, y)['log_loss']) / len(y)
                candidates['poisson'].append({'parameters': {'k': k, 'home_advantage': home, 'scale': scale}, 'log_loss': loss})
    params = {name: min(trials, key=lambda x: x['log_loss'])['parameters'] for name, trials in candidates.items()}
    return {'parameters': params,
            'ratings': {name: elo_ratings(history, p['k'], p['home_advantage']) for name, p in params.items()},
            'mean_goals_per_team': average_goals(history), 'historical': frequency_baseline(history),
            'fit_count': len(history), 'fit_last_date': history[-1]['date'],
            'tuning': {'training_count': len(train), 'training_last_date': train[-1]['date'],
                       'validation_start': '2022-01-01', 'validation_end_exclusive': '2024-01-01',
                       'validation_count': len(validation), 'validation_mean_goals_from_training': mean,
                       'protocol': 'Frozen training-end ratings; no validation updates. Choose minimum validation log loss separately for each model; ties choose first candidate.',
                       'candidates': candidates}}


def forecast(state, home_team, away_team, home_advantage):
    if home_advantage not in (-1, 0, 1):
        raise ValueError('home_advantage must be -1, 0, or 1')
    result = {'uniform': [1 / 3] * 3}
    historical = state['historical']['neutral' if home_advantage == 0 else 'home']
    result['historical'] = list(reversed(historical)) if home_advantage == -1 else list(historical)
    for name in ('elo', 'poisson'):
        ratings, p = state['ratings'][name], state['parameters'][name]
        if home_team not in ratings or away_team not in ratings:
            raise ValueError(f'unknown fixture team; check aliases/history: {home_team}, {away_team}')
        difference = ratings[home_team] - ratings[away_team] + home_advantage * p['home_advantage']
        result[name] = elo_probabilities(difference, p['draw_rate']) if name == 'elo' else poisson_probabilities(difference, state['mean_goals_per_team'], p['scale'])
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_csv(path, fields):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f'{path}: expected CSV header {fields}')
        rows = list(reader)
    if any(None in row or any(v is None or not v.strip() or v != v.strip() for v in row.values()) for row in rows):
        raise ValueError(f'{path}: blank, padded, or malformed fields')
    return rows


def iso_day(value):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('expected YYYY-MM-DD')
    return date.fromisoformat(value).isoformat()


def score(value):
    if not re.fullmatch(r'0|[1-9][0-9]*', value):
        raise ValueError('scores must be nonnegative integers')
    return int(value)


def validate_teams(row):
    if row['home_team'] == row['away_team']:
        raise ValueError('a team cannot play itself')


def load_history(path):
    raw = Path(path).read_bytes()
    rows = read_csv(path, HISTORY_FIELDS)
    selected = []
    before = after = 0
    seen = set()
    for row in rows:
        day = iso_day(row['date'])
        validate_teams(row)
        key = (day, *sorted((row['home_team'], row['away_team'])))
        if key in seen:
            raise ValueError(f'duplicate history match: {key}')
        seen.add(key)
        if row['neutral'] not in ('TRUE', 'FALSE'):
            raise ValueError('neutral must be TRUE or FALSE')
        hs, aws = score(row['home_score']), score(row['away_score'])
        if day >= CUTOFF:
            after += 1
            continue
        if day < TRAIN_START:
            before += 1
            continue
        selected.append(dict(row, date=day, home_score=hs, away_score=aws, neutral=row['neutral'] == 'TRUE'))
    selected.sort(key=lambda r: (r['date'], r['home_team'], r['away_team']))
    if not selected:
        raise ValueError('no eligible history')
    return selected, {'raw_sha256': hashlib.sha256(raw).hexdigest(), 'selected_sha256': digest(selected), 'selected_count': len(selected), 'first_date': selected[0]['date'], 'last_date': selected[-1]['date'], 'excluded_before_start': before, 'excluded_on_or_after_cutoff': after}


def load_fixtures(path):
    rows = read_csv(path, FIXTURE_FIELDS)
    seen = set()
    for row in rows:
        if row['match_id'] in seen:
            raise ValueError('duplicate fixture ID')
        seen.add(row['match_id'])
        validate_teams(row)
        if iso_day(row['date']) < CUTOFF:
            raise ValueError('fixture precedes tournament cutoff')
        if row['home_advantage'] not in ('-1', '0', '1'):
            raise ValueError('invalid fixture home advantage')
        row['home_advantage'] = int(row['home_advantage'])
    if not rows:
        raise ValueError('empty fixtures')
    return sorted(rows, key=lambda row: (row['date'], row['match_id']))


def predict(history_path, fixtures_path):
    history, audit = load_history(history_path)
    fixtures = load_fixtures(fixtures_path)
    state = fit_models(history)
    forecasts = [dict(row, probabilities=forecast(state, row['home_team'], row['away_team'], row['home_advantage'])) for row in fixtures]
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'study_type': 'retrospective pre-tournament-information backtest; not contemporaneous predictions',
            'cutoff_exclusive': CUTOFF, 'history': audit,
            'fixtures_sha256': hashlib.sha256(Path(fixtures_path).read_bytes()).hexdigest(),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'audit': {'ratings_frozen': True, 'outcomes_read': False},
            'fitted': state, 'forecasts': forecasts, 'forecast_sha256': digest(forecasts)}


def evaluate(predictions_path, outcomes_path):
    predictions = json.loads(Path(predictions_path).read_text(encoding='utf-8'))
    forecasts = predictions['forecasts']
    if predictions['forecast_sha256'] != digest(forecasts):
        raise ValueError('forecast integrity hash mismatch')
    outcome_rows = read_csv(outcomes_path, OUTCOME_FIELDS)
    outcomes = {}
    for row in outcome_rows:
        if row['match_id'] in outcomes:
            raise ValueError('duplicate outcome ID')
        outcomes[row['match_id']] = dict(row, home_score=score(row['home_score']), away_score=score(row['away_score']))
    ids = [row['match_id'] for row in forecasts]
    if len(set(ids)) != len(ids) or set(ids) != set(outcomes):
        raise ValueError('forecast and outcome IDs must have an exact one-to-one join')
    labels = [label(outcomes[mid]) for mid in ids]
    probabilities = {name: [row['probabilities'][name] for row in forecasts] for name in ('elo', 'poisson', 'uniform', 'historical')}
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'study_type': predictions['study_type'], 'count': len(ids),
        'predictions_sha256': hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
        'outcomes_sha256': hashlib.sha256(Path(outcomes_path).read_bytes()).hexdigest(),
        'forecast_sha256': predictions['forecast_sha256'],
        'models': {name: metrics(p, labels) for name, p in probabilities.items()},
        'bootstrap': {
            'method': '2000 paired IID match resamples; fixed seed 2026; percentile 95% CI; not adjusted for team/group dependence',
            'interpretation': 'model minus baseline; negative is better for log loss/Brier, positive for accuracy',
            'comparisons': {f'{name}_minus_{baseline}': paired_bootstrap(probabilities[name], probabilities[baseline], labels)
                            for name in ('elo', 'poisson') for baseline in ('uniform', 'historical')}},
        'matches': [dict(row, **{k: v for k, v in outcomes[row['match_id']].items() if k != 'match_id'},
                         actual_class=CLASSES[y]) for row, y in zip(forecasts, labels)],
        'limitations': [
            'Retrospective analysis; no claim forecasts were generated before the event.',
            'Current historical snapshot filtered by match date, not an archived point-in-time database.',
            'Historical goals can include extra time; evaluation is group-stage regulation time only.',
            'Single tournament and repeated teams; bootstrap intervals may understate uncertainty.',
            'No bookmaker odds obtained; no market outperformance or betting profitability established.',
            'Calibration bins are descriptive; no calibration fitting on held-out tournament outcomes.',
            'Uniform argmax ties choose home; its accuracy is an arbitrary tie-break, not random-guess expected accuracy.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prediction = commands.add_parser('predict')
    prediction.add_argument('--history', required=True, type=Path)
    prediction.add_argument('--fixtures', required=True, type=Path)
    prediction.add_argument('--output', required=True, type=Path)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--predictions', required=True, type=Path)
    evaluation.add_argument('--outcomes', required=True, type=Path)
    evaluation.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.history, args.fixtures) if args.command == 'predict' else evaluate(args.predictions, args.outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
