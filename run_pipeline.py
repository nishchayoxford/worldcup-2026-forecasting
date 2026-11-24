"""Reproduce forecasts and evaluation using only the versioned local inputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

import model
import validate_data


MODELS = ('elo', 'poisson', 'historical', 'uniform')
REL_TOL = 1e-10
ABS_TOL = 1e-12
# Only root values are excluded, never nested source/input hashes or dates.
# Timestamps and their predictions-file hash vary per run; code bytes and
# serialized forecast float bits can vary across platforms. Forecast hashes
# are nevertheless verified against each run's own content below.
ROOT_EXCLUSIONS = {
    'predictions.json': {'created_utc', 'code_sha256', 'forecast_sha256'},
    'evaluation.json': {'created_utc', 'predictions_sha256', 'forecast_sha256'},
    'summary.json': {'created_utc', 'code_sha256', 'forecast_sha256'},
}


def compare_values(old, new, path):
    """Compare schemas and discrete values exactly; finite floats tolerantly."""
    if type(old) is not type(new):
        raise ValueError(f'{path}: type changed')
    if isinstance(old, dict):
        if old.keys() != new.keys():
            raise ValueError(f'{path}: keys changed')
        for key in old:
            compare_values(old[key], new[key], f'{path}/{key}')
    elif isinstance(old, list):
        if len(old) != len(new):
            raise ValueError(f'{path}: length changed')
        for index, (a, b) in enumerate(zip(old, new)):
            compare_values(a, b, f'{path}/{index}')
    elif isinstance(old, float):
        if not (math.isfinite(old) and math.isfinite(new)
                and math.isclose(old, new, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
            raise ValueError(f'{path}: numeric value changed or nonfinite')
    elif old != new:
        raise ValueError(f'{path}: value changed')


def compare_json(old, new, artifact):
    if old.keys() != new.keys():
        raise ValueError(f'{artifact}: root keys changed')
    exclusions = ROOT_EXCLUSIONS[artifact]
    for key in exclusions:
        if not isinstance(old.get(key), str) or not isinstance(new.get(key), str):
            raise ValueError(f'{artifact}/{key}: expected execution metadata string')
    compare_values({k: v for k, v in old.items() if k not in exclusions},
                   {k: v for k, v in new.items() if k not in exclusions}, artifact)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def build(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / 'artifacts'
    audit = validate_data.validate(root)
    output.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(root / 'data/history.csv', root / 'data/fixtures.csv')
    save_json(output / 'predictions.json', prediction)
    evaluation = model.evaluate(output / 'predictions.json', root / 'data/outcomes.csv')
    save_json(output / 'evaluation.json', evaluation)
    table = []
    for row in evaluation['matches']:
        flattened = {key: value for key, value in row.items() if key != 'probabilities'}
        for name in MODELS:
            probabilities = row['probabilities'][name]
            flattened.update({f'{name}_p_{label}': p for label, p in zip(model.CLASSES, probabilities)})
            flattened[f'{name}_top_pick'] = model.CLASSES[max(range(3), key=lambda k: probabilities[k])]
        table.append(flattened)
    with (output / 'match_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {
        'study': 'Retrospective pre-tournament-information backtest',
        'created_utc': prediction['created_utc'], 'cutoff_exclusive': model.CUTOFF,
        'evaluated_matches': evaluation['count'], 'data_audit': audit,
        'models': {name: {key: evaluation['models'][name][key]
                         for key in ('accuracy', 'log_loss', 'brier_sum')} | {
                             'correct': sum(row[f'{name}_top_pick'] == row['actual_class'] for row in table)}
                   for name in MODELS},
        'selected_parameters': prediction['fitted']['parameters'],
        'forecast_sha256': prediction['forecast_sha256'],
        'code_sha256': prediction['code_sha256'],
        'metric_conventions': {'brier': 'sum over three classes, then average over matches',
                               'accuracy': 'argmax; ties choose home before draw before away',
                               'log_loss': 'natural logarithm; lower is better'},
    }
    save_json(output / 'summary.json', summary)
    return summary


def read_comparison_csv(path):
    """Parse only probability columns as floats; preserve IDs and other text."""
    probability_fields = [f'{name}_p_{label}' for name in MODELS for label in model.CLASSES]
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or not set(probability_fields) <= set(fields):
            raise ValueError('invalid CSV schema')
        rows = list(reader)
    for row in rows:
        if None in row or None in row.values():
            raise ValueError('invalid CSV row width')
        for field in probability_fields:
            row[field] = float(row[field])
        for name in MODELS:
            model.per_match_metrics([[row[f'{name}_p_{label}'] for label in model.CLASSES]], [0])
    return {'columns': fields, 'rows': rows}


def compare(reference, reproduced):
    """Compare all four artifacts, excluding only explicit root execution metadata."""
    reference, reproduced = Path(reference), Path(reproduced)
    old = json.loads((reference / 'predictions.json').read_text(encoding='utf-8'))
    new = json.loads((reproduced / 'predictions.json').read_text(encoding='utf-8'))
    for artifact in (old, new):
        if artifact['forecast_sha256'] != model.digest(artifact['forecasts']):
            raise ValueError('invalid forecast content hash')
        for row in artifact['forecasts']:
            if set(row['probabilities']) != set(MODELS):
                raise ValueError('unexpected probability model set')
            for probabilities in row['probabilities'].values():
                model.per_match_metrics([probabilities], [0])
    compare_json(old, new, 'predictions.json')
    old_e = json.loads((reference / 'evaluation.json').read_text(encoding='utf-8'))
    new_e = json.loads((reproduced / 'evaluation.json').read_text(encoding='utf-8'))
    for prediction, evaluation in ((old, old_e), (new, new_e)):
        if evaluation['forecast_sha256'] != prediction['forecast_sha256']:
            raise ValueError('evaluation forecast hash does not match its predictions')
        for row in evaluation['matches']:
            if set(row['probabilities']) != set(MODELS):
                raise ValueError('unexpected evaluation probability model set')
            for probabilities in row['probabilities'].values():
                model.per_match_metrics([probabilities], [0])
    compare_json(old_e, new_e, 'evaluation.json')
    old_s = json.loads((reference / 'summary.json').read_text(encoding='utf-8'))
    new_s = json.loads((reproduced / 'summary.json').read_text(encoding='utf-8'))
    for prediction, summary in ((old, old_s), (new, new_s)):
        if summary['forecast_sha256'] != prediction['forecast_sha256']:
            raise ValueError('summary forecast hash does not match its predictions')
    compare_json(old_s, new_s, 'summary.json')
    compare_values(read_comparison_csv(reference / 'match_predictions.csv'),
                   read_comparison_csv(reproduced / 'match_predictions.csv'), 'match_predictions.csv')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='recompute in a temporary directory and compare to committed results')
    parser.add_argument('--plots', action='store_true', help='also regenerate diagnostic figures')
    args = parser.parse_args()
    if args.check and args.plots:
        parser.error('--check and --plots cannot be combined')
    root = Path(__file__).resolve().parent
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build(root, Path(tmp))
            compare(root / 'artifacts', Path(tmp))
        print(f"Reproducibility check passed: {summary['evaluated_matches']} forecasts and all reported scores agree.")
    else:
        summary = build(root)
        if args.plots:
            import plots
            plots.build(root)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
