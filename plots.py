"""Render audited, publication-ready retrospective football diagnostics.

Run ``uv run python plots.py`` from the project root. All chart values come from
analysis.build; no fabricated learning epochs, player data, odds or returns.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

import analysis
import model

PAPER = '#F7F4EC'
INK = '#172E39'
MUTED = '#596C72'
GRID = '#DADFD9'
TEAL = '#087F83'
BLUE = '#3567A6'
ORANGE = '#D66B30'
COLORS = {'elo': TEAL, 'poisson': BLUE, 'uniform': '#9AA4A5', 'historical': '#697B7E'}
NAMES = {'elo': 'Elo–Davidson', 'poisson': 'Elo–Poisson', 'uniform': 'Uniform', 'historical': 'Historical frequency'}
CLASSES = ['Home win', 'Draw', 'Away win']
CMAP = LinearSegmentedColormap.from_list('ivory_teal', [PAPER, '#B8D5CD', TEAL, INK])


def canvas(number, title, subtitle, caption, rows=1, cols=1, size=(12, 6.6), **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=size, layout='constrained', **kwargs)
    fig.get_layout_engine().set(rect=(0.025, 0.12, 0.95, 0.72), h_pad=0.12, w_pad=0.12)
    fig.text(0.04, 0.964, f'WORLD CUP 2026  /  RETROSPECTIVE ANALYSIS                                      {number:02d}',
             color=TEAL, fontsize=9, weight='bold', va='top')
    fig.text(0.04, 0.914, title, color=INK, fontsize=23, weight='bold', va='top')
    fig.text(0.04, 0.862, subtitle, color=MUTED, fontsize=10.5, va='top')
    fig.text(0.04, 0.047, caption, color=MUTED, fontsize=9, va='bottom', linespacing=1.55)
    return fig, axes


def date_values(values):
    return [datetime.fromisoformat(value) for value in values]


def research_design(d):
    c = d['counts']
    caption = ('Ratings are frozen for validation and again for the tournament. The final refit includes validation history.\n'
               'Retrospective reconstruction, not evidence of pre-event publication; current snapshot filtered by match date.')
    fig, axes = canvas(1, 'Separate selection from evaluation',
                       'A chronological protocol with an outcome-free prediction boundary.', caption, cols=2,
                       gridspec_kw={'width_ratios': [1.1, 1]})
    tracks = [('Parameter training', '1990-01-01', '2022-01-01', TEAL, c['training']),
              ('Validation scoring', '2022-01-01', '2024-01-01', BLUE, c['validation']),
              ('Final rating refit', '1990-01-01', model.CUTOFF, '#65998F', c['refit']),
              ('Held-out evaluation', d['audit']['test_first_date'], d['audit']['test_last_date'], ORANGE, c['test'])]
    for ax, zoom in zip(axes, [False, True]):
        for i, (label, start, end, color, count) in enumerate(tracks):
            left, right = mdates.date2num(date_values([start, end]))
            ax.barh(i, max(1, right - left), left=left, height=0.45, color=color)
        ax.set_yticks(range(4), [f'{r[0]}\n{r[4]:,} matches' for r in tracks] if not zoom else [''] * 4)
        ax.invert_yaxis()
        ax.set_ylim(3.8, -0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(1 if zoom else 10))
        ax.set_xlim(date_values(['2021-07-01' if zoom else '1989-01-01', '2027-01-01']))
        ax.set_title('Recent protocol detail' if zoom else 'Full historical window', loc='left', pad=16)
        ax.grid(axis='x')
        ax.axvline(mdates.date2num(date_values([model.CUTOFF])[0]), color=ORANGE, ls='--', lw=1)
    axes[1].text(0.97, 0.98, 'Cutoff: 11 Jun 2026 (exclusive)', transform=axes[1].transAxes,
                 ha='right', va='top', color=ORANGE, fontsize=9)
    return fig, caption


def data_coverage(d):
    caption = ('Counts are eligible international matches, not unique teams. Training begins in 1990; no time weighting is applied.\n'
               '2026 is a partial year ending 10 June. Annual volume is coverage, not a measure of predictive quality.')
    fig, ax = canvas(2, 'The evidence behind the ratings',
                     f"{d['counts']['refit']:,} eligible matches across the pre-cutoff historical window.", caption)
    years = [r['year'] for r in d['annual_coverage']]
    colors = [TEAL if y < 2022 else BLUE if y < 2024 else '#7D9F98' for y in years]
    ax.bar(years, [r['count'] for r in d['annual_coverage']], color=colors, width=0.78)
    ax.set(xlabel='Match year', ylabel='Eligible matches', xlim=(min(years) - 1, max(years) + 1))
    ax.set_xticks(list(range(1990, 2025, 5)) + [2026])
    ax.grid(axis='y')
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in
                       [(TEAL, 'Parameter-training history'), (BLUE, 'Validation window'), ('#7D9F98', 'Later refit history')]],
              loc='upper left', ncols=3, fontsize=9)
    ax.annotate('Partial year', (years[-1], d['annual_coverage'][-1]['count']), xytext=(-36, 30),
                textcoords='offset points', arrowprops={'arrowstyle': '-', 'color': ORANGE}, color=ORANGE, fontsize=9)
    return fig, caption


def rating_trajectories(d):
    caption = ('Eight tournament teams selected by their cutoff Elo rating, not by tournament results. Each line shows post-match-day ratings.\n'
               '1500 initialization; simultaneous-day updates; constant K and home advantage selected on 2022–23 validation.')
    fig, axes = canvas(3, 'How the leading teams reached the cutoff',
                       'Full pre-cutoff Elo histories · selection by final rating · shared vertical scale.', caption,
                       rows=2, cols=4, size=(14, 8.5), sharex=True, sharey=True)
    for ax, team in zip(axes.flat, d['trajectory_selection']):
        series = d['rating_traces']['elo']['teams'][team]
        dates = date_values([r['date'] for r in series])
        values = [r['rating'] for r in series]
        ax.plot(dates, values, color=TEAL, lw=1.3)
        ax.scatter(dates[-1], values[-1], color=ORANGE, s=22, zorder=5)
        ax.axhline(1500, color=MUTED, ls=':', lw=0.8)
        ax.set_title(f'{team}  |  {values[-1]:.0f}', loc='left', fontsize=11)
        ax.grid(axis='y')
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for ax in axes[:, 0]:
        ax.set_ylabel('Elo rating')
    return fig, caption


def team_ranking(d):
    caption = ('Top 16 among the 48 tournament participants; this is not a ranking of every historical national side.\n'
               'Ratings use matches strictly before 11 June 2026. Rating points are not title probabilities or uncertainty estimates.')
    fig, ax = canvas(4, 'The pre-tournament strength hierarchy',
                     'Cutoff Elo ratings of tournament participants, before any group-stage outcome is read.', caption,
                     size=(12, 8.5))
    top = d['ranking'][:16]
    y = np.arange(len(top))
    values = [r['rating'] for r in top]
    ax.hlines(y, 1500, values, color='#B7CDC5', lw=4)
    ax.scatter(values, y, color=TEAL, s=55, zorder=3)
    ax.set_yticks(y, [f"{i + 1:02d}  {r['team']}" for i, r in enumerate(top)])
    ax.invert_yaxis()
    ax.set_xlim(1500, max(values) + 85)
    ax.set_xlabel('Elo rating points (1500 initialization)')
    ax.grid(axis='x')
    for value, row in zip(values, y):
        ax.text(value + 12, row, f'{value:.0f}', va='center', fontsize=10, weight='bold')
    return fig, caption


def parameter_search(d):
    caption = ('Every predeclared candidate is shown: 2 K values × 2 home advantages × 3 draw/scale values per family.\n'
               'Cells are frozen-model 2022–23 validation log loss; lower is better. Orange outlines mark the selected candidate.')
    fig, axes = canvas(5, 'Selection over the complete parameter grid',
                       'No held-out tournament score enters model selection.', caption, rows=2, cols=2, size=(13, 8))
    all_losses = [r['log_loss'] for trials in d['parameter_grid'].values() for r in trials]
    for col, name in enumerate(('elo', 'poisson')):
        field = 'draw_rate' if name == 'elo' else 'scale'
        xs = sorted({r['parameters'][field] for r in d['parameter_grid'][name]})
        for row, k in enumerate((20, 40)):
            ax = axes[row, col]
            values = np.array([[next(r['log_loss'] for r in d['parameter_grid'][name]
                                    if r['parameters'] == {'k': k, 'home_advantage': home, field: x})
                                for x in xs] for home in (0, 60)])
            image = ax.imshow(values, cmap=CMAP.reversed(), vmin=min(all_losses), vmax=max(all_losses), aspect='auto')
            ax.set_xticks(range(3), [str(x) for x in xs])
            ax.set_yticks(range(2), ['Home +0', 'Home +60'])
            ax.set_xlabel('Equal-strength draw probability' if name == 'elo' else 'Elo goal-share scale')
            ax.set_title(f'{NAMES[name]} · K = {k}', loc='left', fontsize=12)
            for i in range(2):
                for j in range(3):
                    normalized = (values[i, j] - min(all_losses)) / (max(all_losses) - min(all_losses))
                    ax.text(j, i, f'{values[i, j]:.4f}', ha='center', va='center', fontsize=12,
                            color=PAPER if normalized < 0.45 else INK)
            selected = d['fitted']['parameters'][name]
            if selected['k'] == k:
                ax.add_patch(Rectangle((xs.index(selected[field]) - 0.48,
                                        (0 if selected['home_advantage'] == 0 else 1) - 0.46),
                                       0.96, 0.92, fill=False, ec=ORANGE, lw=3))
    fig.colorbar(image, ax=axes, label='Validation log loss · lower is better', shrink=0.8, pad=0.025)
    return fig, caption


def validation_path(d):
    caption = ('Cumulative mean loss over chronological validation matches, using fixed training-end ratings throughout.\n'
               'These are validation scoring paths, not epochs, optimization traces or online rating updates; the window selected the models.')
    fig, ax = canvas(6, 'Validation performance accumulates over time',
                     f"Selected candidates scored on {d['counts']['validation']:,} matches in 2022–23, with no rating updates.", caption)
    for name in ('elo', 'poisson'):
        rows = d['validation'][name]['matches']
        values = [r['cumulative_log_loss'] for r in rows]
        ax.plot(date_values([r['date'] for r in rows]), values, color=COLORS[name], lw=1.8,
                label=f'{NAMES[name]} · final {values[-1]:.4f}')
    ax.axhline(np.log(3), color=MUTED, ls='--', lw=1, label='Uniform reference · ln(3)')
    ax.set(xlabel='Validation match date', ylabel='Cumulative mean log loss')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(axis='y')
    ax.legend(loc='upper right', fontsize=10)
    return fig, caption


def model_comparison(d):
    caption = ('Log loss uses natural logarithms; Brier is the sum over three classes, not the class-average convention.\n'
               'Accuracy is argmax classification. Uniform ties choose home; its accuracy is not random-guess expected accuracy.')
    fig, axes = canvas(7, 'Held-out scores, including simple baselines',
                       f"One frozen forecast per fixture · {d['counts']['test']} group-stage matches · lower proper scores are better.", caption,
                       cols=3, size=(14, 6.8))
    names = ['elo', 'poisson', 'historical', 'uniform']
    for ax, metric, title in zip(axes, ['log_loss', 'brier_sum', 'accuracy'],
                                  ['Log loss ↓', 'Brier sum ↓', 'Accuracy ↑']):
        values = [d['metrics'][name][metric] for name in names]
        ax.barh(range(4), values, color=[COLORS[name] for name in names], height=0.55)
        ax.set_yticks(range(4), [NAMES[name] for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.27)
        ax.set_title(title, loc='left')
        ax.grid(axis='x')
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.025, i, f'{value:.1%}' if metric == 'accuracy' else f'{value:.3f}',
                    va='center', weight='bold', fontsize=10)
        if metric == 'accuracy':
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    return fig, caption


def calibration(d):
    caption = ('Ten equal-width probability bins; labels show bin counts n. Empty bins are omitted. Diagonal indicates exact calibration.\n'
               '95% Wilson binomial intervals are unadjusted for repeated-team/group dependence; no recalibration is fitted to test outcomes.')
    fig, axes = canvas(8, 'Calibration: probability versus observed frequency',
                       'Top-label correctness and one-versus-rest class reliability, shown separately by model.', caption,
                       rows=2, cols=4, size=(16, 9), sharex=True, sharey=True)
    keys = ['top_label', *model.CLASSES]
    for row, name in enumerate(('elo', 'poisson')):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            bins = [b for b in d['calibration'][name][key] if b['count']]
            x = np.array([b['mean_probability'] for b in bins])
            y = np.array([b['observed_frequency'] for b in bins])
            limits = np.array([b['wilson_ci95'] for b in bins])
            ax.plot([0, 1], [0, 1], color=MUTED, ls='--', lw=0.8)
            ax.errorbar(x, y, yerr=np.maximum(0, np.array([y - limits[:, 0], limits[:, 1] - y])),
                        fmt='o', markersize=4, color=COLORS[name], capsize=3, lw=1.1)
            for index, b in enumerate(bins):
                ax.annotate(str(b['count']), (b['mean_probability'], b['observed_frequency']),
                            xytext=(4, 7 if index % 2 == 0 else -14), textcoords='offset points',
                            fontsize=8, color=INK, bbox={'facecolor': PAPER, 'edgecolor': 'none', 'pad': 0.4})
            ax.set_title(('Top-label correctness' if col == 0 else CLASSES[col - 1]), fontsize=11, loc='left')
            ax.set(xlim=(-0.04, 1.04), ylim=(-0.06, 1.08), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
            ax.grid(alpha=0.55)
            if col == 0:
                ax.set_ylabel(f'{NAMES[name]}\nObserved frequency')
            if row == 1:
                ax.set_xlabel('Mean forecast probability')
    return fig, caption


def confusion_matrices(d):
    caption = ('Counts, not row-normalized percentages. Rows are actual classes; columns are argmax predictions.\n'
               'Nonzero draw probabilities do not imply draw picks: the draw column is retained even when every entry is zero.')
    fig, axes = canvas(9, 'Where the classification errors occur',
                       'All three classes remain visible, including draws omitted by the argmax decision rule.', caption,
                       cols=2, size=(12, 6.8))
    maximum = max(max(row) for matrix in d['confusion'].values() for row in matrix)
    for ax, name in zip(axes, ('elo', 'poisson')):
        matrix = np.array(d['confusion'][name])
        image = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=maximum)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', fontsize=22, weight='bold',
                    color=PAPER if value > maximum * 0.55 else INK)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set(xlabel='Predicted class', ylabel='Actual class')
        ax.set_title(f"{NAMES[name]} · {matrix[:, 1].sum()} draw picks", loc='left', fontsize=12)
    fig.colorbar(image, ax=axes, label='Match count', shrink=0.85, pad=0.03)
    return fig, caption


def match_probabilities(d):
    caption = ('All held-out fixtures, sorted by group, date and match ID. H / D / A = listed home win / draw / listed away win.\n'
               'Numbers are forecast percentages; orange outlines mark the realized class. Home/away listing is not always venue advantage.')
    fig, axes = canvas(10, 'Every match, every outcome probability',
                       'A complete forecast audit · observed results are annotations, never predictive inputs.', caption,
                       cols=2, size=(16, 23), sharey=True)
    fig.get_layout_engine().set(rect=(0.015, 0.055, 0.97, 0.86), w_pad=0.15)
    # Use a compact header on a tall figure without enlarging the font.
    fig.texts[0].set_y(0.985)
    fig.texts[1].set_y(0.965)
    fig.texts[2].set_y(0.943)
    fig.texts[3].set_y(0.015)
    rows = sorted(d['matches'], key=lambda r: (r['group'], r['date'], r['match_id']))
    labels = [f"{r['group'].replace('Group ', '')}  {r['date'][5:]}  {r['home_team']} – {r['away_team']}  {r['home_score']}:{r['away_score']}" for r in rows]
    for ax, name in zip(axes, ('elo', 'poisson')):
        values = np.array([r['probabilities'][name] for r in rows])
        ax.imshow(values, cmap=CMAP, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(3), ['H', 'D', 'A'])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), labels, fontsize=8.4)
        ax.set_title(NAMES[name], fontsize=14, loc='left', pad=30)
        for i, row in enumerate(rows):
            for j in range(3):
                ax.text(j, i, f'{values[i, j]:.0%}', ha='center', va='center', fontsize=8.7,
                        color=PAPER if values[i, j] > 0.56 else INK)
            ax.add_patch(Rectangle((model.CLASSES.index(row['actual_class']) - 0.46, i - 0.44),
                                   0.92, 0.88, fill=False, ec=ORANGE, lw=1.4))
            if i and row['group'] != rows[i - 1]['group']:
                ax.axhline(i - 0.5, color=INK, lw=1.3)
        ax.tick_params(axis='y', length=0)
    axes[1].tick_params(axis='y', labelleft=False)
    return fig, caption


def group_performance(d):
    caption = ('Arithmetic mean log loss within each group; dotted lines show model-wide averages. No shrinkage or group-specific fitting.\n'
               'Small groups make apparent differences noisy. Group breakdowns are descriptive, not independent validation sets.')
    fig, ax = canvas(11, 'Performance varies across tournament groups',
                     'The overall average conceals substantial local variation.', caption)
    rows = d['group_scores']
    positions = np.arange(len(rows))
    for name, offset in [('elo', -0.12), ('poisson', 0.12)]:
        values = [r[name] for r in rows]
        ax.scatter(positions + offset, values, color=COLORS[name], s=55, label=NAMES[name], zorder=4)
        ax.axhline(d['metrics'][name]['log_loss'], color=COLORS[name], ls=':', lw=1)
    for x, row in zip(positions, rows):
        ax.plot([x - 0.12, x + 0.12], [row['elo'], row['poisson']], color='#A5B7B4', lw=1.5)
    ax.axhline(np.log(3), color=MUTED, ls='--', lw=1, label='Uniform reference')
    ax.set_xticks(positions, [f"{r['group']}\nn={r['count']}" for r in rows], fontsize=9)
    ax.set(ylabel='Mean group log loss · lower is better', xlabel='Tournament group')
    ax.grid(axis='y')
    ax.legend(loc='upper left', ncols=3, fontsize=9)

    return fig, caption


def uncertainty(d):
    caption = ('Paired log-loss differences; 4,000 resamples, seed 2026, percentile 95% intervals. Negative values favor the first model.\n'
               'Group resampling keeps each group intact; only 12 clusters. Conditional on fixed models; not team/network robust or evidence of betting edge.')
    fig, ax = canvas(12, 'Apparent advantages carry uncertainty',
                     'Match-IID and whole-group resampling provide different views of the same paired contrasts.', caption,
                     size=(13, 7.5))
    labels = []
    for i, (key, result) in enumerate(d['comparisons'].items()):
        name, reference = key.split('_minus_')
        labels.append(f'{NAMES[name]}\nminus {NAMES[reference]}')
        for method, offset, marker, alpha in [('iid', -0.12, 'o', 0.45), ('group_cluster', 0.12, 's', 1)]:
            lo, hi = result[method]['ci95']
            ax.plot([lo, hi], [i + offset, i + offset], color=COLORS[name], alpha=alpha, lw=2)
            ax.scatter(result['difference'], i + offset, marker=marker, color=COLORS[name], alpha=alpha, s=40, zorder=4)
            ax.plot([lo, lo, np.nan, hi, hi], [i + offset - 0.05, i + offset + 0.05, np.nan, i + offset - 0.05, i + offset + 0.05],
                    color=COLORS[name], alpha=alpha, lw=1)
    ax.axvline(0, color=ORANGE, ls='--', lw=1.3)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean log-loss difference · first model minus reference')
    ax.grid(axis='x')
    ax.legend(handles=[Line2D([0], [0], color=INK, alpha=0.45, marker='o', label='IID matches'),
                       Line2D([0], [0], color=INK, marker='s', label='Whole groups')], loc='upper right', fontsize=10)
    return fig, caption


def forecast_errors(d):
    caption = ('For each model, the eight incorrect argmax predictions with highest confidence; ties use date and match ID.\n'
               'Labels show actual score. This post-hoc diagnostic does not alter parameters or recalibrate forecasts; high confidence is not certainty.')
    fig, axes = canvas(13, 'The most confident mistakes',
                       'Contrast probability assigned to the chosen class with probability assigned to the realized class.', caption,
                       cols=2, size=(16, 10))
    for ax, name in zip(axes, ('elo', 'poisson')):
        rows = [r for r in d['matches'] if np.argmax(r['probabilities'][name]) != model.CLASSES.index(r['actual_class'])]
        rows.sort(key=lambda r: (-max(r['probabilities'][name]), r['date'], r['match_id']))
        rows = rows[:8]
        for i, row in enumerate(rows):
            p = row['probabilities'][name]
            confidence, realized = max(p), p[model.CLASSES.index(row['actual_class'])]
            ax.plot([realized, confidence], [i, i], color='#C3CBC6', lw=3)
            ax.scatter(confidence, i, color=ORANGE, s=45, zorder=4)
            ax.scatter(realized, i, color=COLORS[name], s=45, zorder=4)
            ax.text(confidence + 0.025, i, f'{confidence:.0%}', va='center', fontsize=9, color=ORANGE)
        ax.set_yticks(range(len(rows)), [f"{r['home_team']} – {r['away_team']}\n{r['date'][5:]} · actual {r['home_score']}:{r['away_score']}" for r in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(NAMES[name], loc='left', fontsize=13)
        ax.set_xlabel('Forecast probability')
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='x')
        ax.legend(handles=[Line2D([0], [0], color=ORANGE, marker='o', ls='', label='Chosen class'),
                           Line2D([0], [0], color=COLORS[name], marker='o', ls='', label='Realized class')],
                  loc='lower right', fontsize=9)
    return fig, caption


FIGURES = [
    ('01_research_design', research_design, ['counts', 'audit']),
    ('02_data_coverage', data_coverage, ['annual_coverage']),
    ('03_rating_trajectories', rating_trajectories, ['rating_traces.elo', 'trajectory_selection']),
    ('04_team_ranking', team_ranking, ['ranking']),
    ('05_parameter_search', parameter_search, ['parameter_grid', 'fitted.parameters']),
    ('06_validation_path', validation_path, ['validation']),
    ('07_model_comparison', model_comparison, ['metrics']),
    ('08_calibration', calibration, ['calibration']),
    ('09_confusion_matrices', confusion_matrices, ['confusion']),
    ('10_match_probabilities', match_probabilities, ['matches']),
    ('11_group_performance', group_performance, ['group_scores', 'metrics']),
    ('12_uncertainty', uncertainty, ['comparisons', 'uncertainty']),
    ('13_forecast_errors', forecast_errors, ['matches']),
]


def build(root):
    """Regenerate audited diagnostics and the complete PNG/SVG figure manifest."""
    root = Path(root)
    d = analysis.build(root)
    folder = root / 'docs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
             'axes.labelcolor': INK, 'axes.titlecolor': INK, 'axes.facecolor': PAPER,
             'figure.facecolor': PAPER, 'savefig.facecolor': PAPER, 'axes.edgecolor': GRID,
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
             'axes.spines.bottom': False, 'axes.axisbelow': True, 'axes.titleweight': 'bold',
             'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
             'grid.color': GRID, 'grid.linewidth': 0.7, 'legend.frameon': False,
             'svg.fonttype': 'none', 'svg.hashsalt': 'worldcup-2026-forecasting-v1'}
    with plt.rc_context(style):
        for stem, draw, keys in FIGURES:
            fig, caption = draw(d)
            title = fig.texts[1].get_text()
            paths, hashes = {}, {}
            for extension in ('png', 'svg'):
                relative = f'docs/figures/{stem}.{extension}'
                metadata = {'Title': title, 'Description': caption, 'Creator': 'World Cup 2026 forecasting'}
                if extension == 'svg':
                    metadata['Date'] = None
                else:
                    metadata['Software'] = 'Matplotlib'
                fig.savefig(root / relative, dpi=180, metadata=metadata)
                paths[extension] = relative
                hashes[extension] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            entries.append({'id': stem, 'title': title, 'caption': caption, 'data_keys': keys,
                            **paths, 'sha256': hashes, 'size_inches': list(fig.get_size_inches())})
            plt.close(fig)
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'study_type': d['study_type'], 'test_match_count': d['counts']['test'],
                'figure_count': len(entries), 'file_count': len(entries) * 2,
                'diagnostics_sha256': hashlib.sha256((root / 'artifacts/diagnostics.json').read_bytes()).hexdigest(),
                'plots_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
                'reproduce': 'uv run python plots.py', 'figures': entries}
    (root / 'artifacts/figure_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = build(args.root)
    print(f"Rendered {result['figure_count']} figures ({result['file_count']} PNG/SVG files); artifacts/figure_manifest.json")
