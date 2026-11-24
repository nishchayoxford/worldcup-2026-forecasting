# Methodology and model card

## Research question

Can simple, interpretable team-strength models produce useful three-way probabilities for international football matches, when evaluated on a tournament excluded from fitting and model selection?

**Target:** regulation-time home win, draw or away win. **Test set:** the 72 group-stage matches of the men's 2026 World Cup. “Home” denotes the first-listed team, not necessarily the host.

This is a retrospective study with a pre-tournament information cutoff, not a claim of forecasts published before the event. The repository stores actual execution timestamps and versioned source snapshots.

## 1. Temporal protocol

| Stage | Match dates | What is allowed |
|---|---|---|
| Initial fit | 1990–2021 | Estimate team ratings, scoring intensity and baseline frequencies |
| Parameter selection | 2022–2023 | Score frozen training-end ratings; choose grid settings by mean log loss |
| Final refit | 1990–10 June 2026 | Rebuild ratings and nuisance estimates using the selected settings |
| Test | 2026 World Cup group stage | Produce all forecasts with frozen ratings, then evaluate once |

The initial fit has 27,728 matches. Validation has 2,034 matches. The final refit has 32,325 matches. Parameter selection never sees 2026 tournament outcomes. Reusing validation matches in the final refit is intentional: selection is complete before the held-out tournament is scored.

Frozen two-year validation is conservative but differs from the shorter tournament horizon. It was chosen for a transparent first study, not because it maximizes test performance. A future extension should compare multiple historical tournament holdouts and nested rolling validation.

The model has **no neural-network epochs**. Its training diagnostics are chronological rating updates, a finite hyperparameter grid and cumulative validation scoring—not invented gradient-descent learning curves.

## 2. Historical rating updates

Each team starts at 1500 Elo points. Let $R_h$ and $R_a$ be pre-match ratings and $H$ the home bonus, applied only on non-neutral historical fixtures:

$$d = R_h - R_a + H,$$

$$E_h = \frac{1}{1+10^{-d/400}}, \qquad \Delta R_h = K(S_h-E_h), \qquad \Delta R_a=-\Delta R_h,$$

where $S_h$ is 1 for a home win, 0.5 for a draw and 0 for an away win. Matches sharing a date use the same day's opening ratings, then apply updates together, avoiding arbitrary within-day ordering.

All competitions have equal update weight. Constant $K$ gives ratings an implicit adaptation rate, but there is no explicit time decay, margin-of-victory multiplier, player availability or team-specific attack/defence parameter.

For target fixtures the venue indicator is $v\in\{-1,0,1\}$, so $d=R_h-R_a+vH$. A co-host receives an advantage only when playing in its own country, not anywhere in the co-hosted tournament.

## 3. Elo with a draw term

Use a Davidson-style extension to convert ratings into three probabilities. Define

$$r=10^{d/800},\qquad \nu=\frac{2q}{1-q},\qquad Z=r+\nu+r^{-1}.$$

Then

$$P(H)=\frac rZ,\qquad P(D)=\frac\nu Z,\qquad P(A)=\frac{r^{-1}}Z.$$

The hyperparameter $q$ is the draw probability for equally rated teams. Draw probability decreases as the strength gap increases. This probability mapping is a separate forecasting layer over conventional binary-score Elo updates; the updates are not the maximum-likelihood fit of a Davidson model.

## 4. Elo-informed independent Poisson

Let $\bar g$ be historical goals per team per match and $s$ the strength scale. Define

$$u=\exp(sd/400),\qquad \lambda_h=2\bar g\frac{u}{1+u},\qquad \lambda_a=2\bar g\frac1{1+u}.$$

Assume independent scores $G_h\sim\operatorname{Poisson}(\lambda_h)$ and $G_a\sim\operatorname{Poisson}(\lambda_a)$. Sum score-cell probabilities over $G_h>G_a$, equality and $G_h<G_a$ to obtain the three outcomes. Numerical summation retains each marginal's mass to within $10^{-13}$, followed by normalization.

This is a deliberately small **rating-to-goals model** with constant total intensity $2\bar g$. It is **not** a fitted attack/defence, bivariate-Poisson, xG or Dixon–Coles model. Independent scores and a shared total intensity are strong assumptions.

## 5. Selection grid

Both families try $K\in\{20,40\}$ and $H\in\{0,60\}$.

- Elo draws: $q\in\{0.22,0.28,0.34\}$.
- Poisson scale: $s\in\{0.75,1.25,1.75\}$.

Each family has 12 candidates. Choose the lowest 2022–2023 log loss; exact ties select the first candidate. Both families remain in the final report rather than promoting the lower-loss test model as a newly selected winner.

Selected settings: **$K=40$, $H=60$** for both; **$q=0.22$** for Elo and **$s=1.75$** for Poisson. Some optima lie on grid boundaries. That motivates a predeclared wider-grid experiment on separate historical holdouts—not post-hoc expansion using the 2026 results.

## 6. Baselines and evaluation

**Uniform:** one-third probability for each outcome.

**Historical frequencies:** add-one-smoothed outcomes stratified by neutral/non-neutral venue. Neutral home/away frequencies are symmetrized because first-listed status is largely arbitrary. Probabilities reverse when the second-listed team has venue advantage.

Primary scores are proper scoring rules:

$$\operatorname{LogLoss}=-\frac1N\sum_i\log p_{i,y_i},$$

$$\operatorname{Brier}=\frac1N\sum_i\sum_{k=1}^3(p_{ik}-\mathbf1[y_i=k])^2.$$

The Brier score is **summed over classes**, not divided by three. Lower is better for both. Accuracy uses argmax, with first-class tie-breaking. Uniform baseline accuracy therefore reflects arbitrary ties, not the expected accuracy of random guessing.

Calibration is summarized using ten equal-width bins for top-label confidence and each one-vs-rest outcome. These are descriptive evaluations; no calibrator is trained on tournament results. Empty bins carry no empirical estimate, and small bins should not be overinterpreted.

## 7. Uncertainty

The base evaluator uses 2,000 paired match-level bootstrap resamples, seed 2026. A comparison is **model loss minus baseline loss**; negative values favor the model. Pairing preserves the match-specific relationship between model and baseline errors.

Matches share teams and group conditions, so independent-match intervals can be too narrow. The diagnostic figure compares 4,000 paired IID resamples with 4,000 whole-group resamples, using NumPy's random generator and seed 2026. Grouping preserves dependence within a group but offers only 12 sampling units and does not remove all tournament-wide dependence. This separate sensitivity calculation need not have identical endpoints to the base evaluator's 2,000 resamples. Intervals condition on fixed models and omit parameter-selection uncertainty; they are not evidence of market efficiency violations.

## 8. Main limitations

1. **Data vintage:** current historical snapshot filtered by match date, not a point-in-time database. Fixtures are reconstructed from completed-event metadata. Programmatic outcome separation is not proof of researcher blindness.
2. **Training labels:** historical source scores include extra time; group-stage test labels are regulation-time results. Shootout goals are excluded.
3. **Small holdout:** 72 games, 48 teams and one tournament do not establish broad generalization.
4. **Draws:** both models assign positive draw probabilities but select no draws as their argmax in this run. Probability scores and class calibration matter more than headline accuracy.
5. **Model simplicity:** no player/injury information, lineups, xG, competition weighting, explicit temporal decay or correlated low-score adjustment.
6. **No market test:** no timestamped bookmaker odds, spreads, execution costs or betting returns. Beating simple baselines is not beating a sportsbook.
7. **Scope:** group stage only. A pre-tournament knockout study would need to simulate advancement and unknown pairings rather than use the realized bracket.

## 9. Reproducibility and audit

`validate_data.py` checks byte hashes, schema and tournament completeness. `model.py predict` accepts history and score-free fixtures only. `model.py evaluate` reads saved forecasts and outcomes separately, checks the forecast content digest and requires an exact one-to-one match-ID join.

The low-level model CLI intentionally accepts smaller well-formed fixture sets for testing and reuse; it does not independently enforce the 72-match tournament design. Use `run_pipeline.py` for this study: it validates the complete group-stage inputs before fitting. The regulation-time interpretation applies to the supplied group-stage labels, not arbitrary replacement data passed to the generic CLI.

`run_pipeline.py --check` recomputes the four core artifacts in a temporary directory and compares all deterministic contents of `predictions.json`, `evaluation.json`, `summary.json` and `match_predictions.csv`, including calibration, bootstrap intervals and actual outcomes. Discrete values, schemas and input hashes compare exactly; floats use `rel_tol=1e-10` and `abs_tol=1e-12`. Only explicitly listed root-level execution timestamps and execution/serialization-dependent hashes are excluded from cross-run comparison. Each forecast content hash is still validated against its own contents. Exact serialized hashes are integrity checks for a particular artifact, not a promise of identical floating-point bytes on every Python version and not proof of historical publication.
