"""Validate the versioned inputs before fitting or reproducing the study."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import model


def validate(root):
    root = Path(root)
    manifest = json.loads((root / 'data/manifest.json').read_text(encoding='utf-8'))
    expected_files = {'history.csv', 'fixtures.csv', 'outcomes.csv'}
    if set(manifest['files']) != expected_files:
        raise ValueError('manifest must describe exactly the three expected research inputs')
    for name in sorted(expected_files):
        payload = (root / 'data' / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest['files'][name]['sha256']:
            raise ValueError(f'integrity check failed: {name}')
    history, audit = model.load_history(root / 'data/history.csv')
    fixtures = model.load_fixtures(root / 'data/fixtures.csv')
    if any(not '2026-06-11' <= row['date'] <= '2026-06-28' for row in fixtures):
        raise ValueError('fixture outside the World Cup group-stage date window')
    targets = model.read_csv(root / 'data/outcomes.csv', model.OUTCOME_FIELDS)
    sizes = {'history.csv': len(history), 'fixtures.csv': len(fixtures), 'outcomes.csv': len(targets)}
    if any(size != manifest['files'][name]['rows'] for name, size in sizes.items()):
        raise ValueError('input row count disagrees with manifest')
    if audit['excluded_before_start'] or audit['excluded_on_or_after_cutoff']:
        raise ValueError('versioned history contains out-of-window records')
    fixture_ids = {r['match_id'] for r in fixtures}
    target_ids = [r['match_id'] for r in targets]
    if len(target_ids) != len(set(target_ids)) or set(target_ids) != fixture_ids:
        raise ValueError('targets must have an exact one-to-one join with fixtures')
    for row in targets:
        model.score(row['home_score'])
        model.score(row['away_score'])
    groups = defaultdict(list)
    for fixture in fixtures:
        groups[fixture['group']].append(fixture)
    appearances = Counter(r[key] for r in fixtures for key in ('home_team', 'away_team'))
    if len(fixtures) != 72 or len(groups) != 12 or len(appearances) != 48:
        raise ValueError('expected a complete 48-team, 12-group, 72-match group stage')
    if set(appearances.values()) != {3}:
        raise ValueError('each team must appear exactly three times')
    group_teams = []
    for group, rows in groups.items():
        teams = {r[k] for r in rows for k in ('home_team', 'away_team')}
        pairs = [tuple(sorted((r['home_team'], r['away_team']))) for r in rows]
        if len(teams) != 4 or len(rows) != 6 or set(pairs) != set(combinations(sorted(teams), 2)):
            raise ValueError(f'invalid round-robin schedule: {group}')
        group_teams.extend(teams)
    if len(set(group_teams)) != len(group_teams):
        raise ValueError('a team appears in more than one group')
    known = {r[k] for r in history for k in ('home_team', 'away_team')}
    if set(appearances) - known:
        raise ValueError('fixture team missing from historical data')
    return {'history_matches': len(history), 'history_first_date': audit['first_date'],
            'history_last_date': audit['last_date'], 'fixtures': len(fixtures),
            'teams': len(appearances), 'groups': len(groups), 'target_ids_match': True,
            'data_manifest_sha256': hashlib.sha256((root / 'data/manifest.json').read_bytes()).hexdigest(),
            'input_hashes': {name: record['sha256'] for name, record in manifest['files'].items()}}


if __name__ == '__main__':
    print(json.dumps(validate(Path(__file__).resolve().parent), indent=2))
