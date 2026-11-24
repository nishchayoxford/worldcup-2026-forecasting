# Research artifacts

These files are computed outputs, not hand-authored performance claims. Rebuild the core artifacts with `uv run python run_pipeline.py`; add `--plots` to rebuild the diagnostic figures as well.

| Artifact | Contents |
|---|---|
| [summary.json](summary.json) | Compact headline metrics, selected parameters, counts and input integrity hashes |
| [match_predictions.csv](match_predictions.csv) | All 72 fixtures, four model probability vectors, actual goals and argmax picks |
| [predictions.json](predictions.json) | Frozen forecasts, full team ratings, all validation candidate scores and provenance |
| [evaluation.json](evaluation.json) | Proper scores, class/top-label calibration, paired bootstrap intervals and matched outcomes |
| [diagnostics.json](diagnostics.json) | Complete historical rating traces, validation paths, group-level diagnostics and sensitivity calculations; compact JSON to limit repository size |
| [figure_manifest.json](figure_manifest.json) | Every chart's title, caption, source keys, PNG/SVG paths, hashes and rendering environment |

## Probability and score conventions

- Class order: **home win, draw, away win**.
- “Home” means the first-listed team; venue advantage is separate.
- Log loss uses natural logarithms. Lower is better.
- Multiclass Brier is summed across the three classes, then averaged over games. Lower is better.
- Accuracy uses the highest-probability class; exact ties choose home before draw before away.
- Bootstrap contrasts are **first model minus reference**. Negative log-loss differences favor the first model.
- `evaluation.json` uses 2,000 paired IID resamples. Diagnostic charts separately use 4,000 paired IID/whole-group resamples. Their interval endpoints need not be identical.

## What hashes and timestamps mean

Input hashes identify the precise versioned CSV bytes. Forecast hashes check the integrity of serialized probabilities and fixture metadata. Actual UTC execution times are retained. Neither a hash nor a retrospective information cutoff proves that forecasts were published before the tournament.

Small floating-point differences can occur across Python versions and operating systems. Reproducibility checks compare numerical results with declared tolerances rather than requiring cross-platform byte equality. Run:

```bash
uv run python run_pipeline.py --check
```

The versioned data can be audited independently:

```bash
uv run python validate_data.py
```

See [methodology](../docs/METHODOLOGY.md) for the study design and uncertainty limitations, and [data provenance](../data/README.md) for licensing and source limitations.
