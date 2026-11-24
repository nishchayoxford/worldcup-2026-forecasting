# Data dictionary and provenance

The repository includes the small, versioned research inputs needed to reproduce the study **offline**. It does not redistribute raw website pages, API responses, logos, photographs or editorial text.

## Sources and licenses

**Historical results and target scores:** [martj42/international_results](https://github.com/martj42/international_results), released under **CC0 1.0**. The upstream license is preserved in [LICENSE-CC0.txt](LICENSE-CC0.txt). The source CSV URL, actual retrieval timestamp, original response SHA-256 and derivative file hashes are recorded in [manifest.json](manifest.json).

- Upstream CSV: <https://raw.githubusercontent.com/martj42/international_results/master/results.csv>
- Upstream methodology: <https://github.com/martj42/international_results/blob/master/README.md>
- Eligible training records: 1 January 1990 inclusive to 11 June 2026 exclusive.
- Actual retained coverage: 12 January 1990–10 June 2026, 32,325 matches.
- Target goals are joined from that same CC0 source's 2026 World Cup records by unique team pair and date within one day, accommodating UTC/local-date differences. All 72 match score pairs were checked against the original scoreboard-derived target set.

**Fixture facts:** the minimal score-free schedule contains match ID, UTC date, group, team names and venue-based home advantage. It was reconstructed from [ESPN's public World Cup scoreboard](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260611&limit=1000). Match IDs are retained only as join identifiers. No raw ESPN payloads, commentary, team logos, standings or player data are included. These factual fixture annotations are distinguished from the CC0 historical dataset; the repository does not claim ownership of third-party trademarks or impose a new license on third-party source material.

## Files

| File | Rows | Role |
|---|---:|---|
| `history.csv` | 32,325 | Training, chronological validation and final refit |
| `fixtures.csv` | 72 | Score-free group-stage prediction input |
| `outcomes.csv` | 72 | Held-out labels read only by evaluation |
| `manifest.json` | — | Provenance, counts and SHA-256 integrity checks |

### `history.csv`

| Field | Meaning |
|---|---|
| `date` | Source match date, `YYYY-MM-DD` |
| `home_team`, `away_team` | Harmonized national-team names |
| `home_score`, `away_score` | Goals **including extra time if played**, excluding penalty shootouts |
| `tournament` | Competition label, including friendlies and non-FIFA internationals |
| `city`, `country` | Venue metadata from source; not predictors in this implementation |
| `neutral` | `TRUE`/`FALSE`; controls historical home advantage |

### `fixtures.csv`

| Field | Meaning |
|---|---|
| `match_id` | Stable match-level join identifier |
| `date` | UTC fixture date |
| `group` | Group A–L |
| `home_team`, `away_team` | First- and second-listed team; neither implies host advantage |
| `home_advantage` | `1` if first-listed team plays in its own country, `-1` if second-listed team does, otherwise `0` |

### `outcomes.csv`

`match_id,home_score,away_score`. Group-stage games have no extra time, so these are regulation-time outcomes.

## Harmonization and quality controls

Aliases are resolved explicitly: Bosnia-Herzegovina → Bosnia and Herzegovina; Czechia → Czech Republic; Congo DR → DR Congo; Türkiye → Turkey; USA → United States.

`validate_data.py` verifies hashes and schema, the exclusive training cutoff, fixture dates, unique IDs, exact target joins, 48 distinct teams, 12 disjoint groups, each group's complete six-match round robin and exactly three appearances per team. Every fixture team must exist in historical data.

## What the data cannot establish

This is a **current source snapshot filtered retrospectively by match date**, not a frozen database retrieved on 10 June 2026. Later source corrections are possible. Fixtures were reconstructed from completed-event metadata, although group-stage pairings were knowable before kickoff. Structural checks do not establish exhaustive completeness or independent confirmation of every historical match. The source's extra-time convention creates a training-label mismatch with regulation-time targets.
