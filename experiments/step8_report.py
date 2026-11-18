"""Step 8 - assemble Tables 4-11 and Figure 3 in the layout of the manuscript.

Learned methods are reported as mean +- standard deviation over the five seeds; fixed scores are
deterministic.  Output: results/RESULTS.md, results/tables/*.csv, results/figures/figure3.{png,pdf}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_json, mean_sd, parse_args, path  # noqa: E402
from lawsr import config  # noqa: E402

DEC = {'R2': 3, 'MSE': 4, 'RMSE': 3, 'MAE': 3, 'Pearson': 3, 'Spearman': 3, 'AUC': 3, 'AP': 3,
       'Within-file Spearman': 3, 'Top-1 (%)': 1, 'Top-10 (%)': 1, 'MRR': 3, 'Same-file P@10 (%)': 1}


def fmt(v, d, sd=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    s = f'{v:.{d}f}'.replace('-', '−')
    if sd is not None:
        s += f' ± {sd:.{d}f}'
    return s


def cell(entry, key, d=None):
    """entry: fixed metrics dict, or {'per_seed': {...}} for learned methods."""
    d = DEC.get(key, 3) if d is None else d
    if 'per_seed' in entry:
        m, sd = mean_sd([v[key] for v in entry['per_seed'].values()])
        return fmt(m, d, sd), m, sd
    return fmt(entry[key], d), entry[key], None


def md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(lines)


def table(rows, cols, name, args, title, note=None):
    df = pd.DataFrame([[r[0]] + [r[1][c][0] for c in cols] for r in rows], columns=[rows_label(name)] + cols)
    raw = pd.DataFrame([{**{'method': r[0]}, **{c: r[1][c][1] for c in cols}, **{f'{c} sd': r[1][c][2] for c in cols}}
                        for r in rows])
    raw.to_csv(path(args, 'tables', f'{name}.csv'), index=False)
    text = f'### {title}\n\n' + md(df) + '\n'
    if note:
        text += f'\n{note}\n'
    return text


def rows_label(name):
    return {'table4': 'Method', 'table6': 'Input scaling', 'table7': 'Scoring method', 'table8': 'Condition',
            'table10': 'Ranking before validation', 'table11a': 'Input configuration',
            'table11b': 'Model and test condition'}.get(name, 'Row')


def main():
    args = parse_args(__doc__)
    sel = load_json('metrics', 'selection.json', args=args)
    lrn = load_json('metrics', 'learnability.json', args=args)
    pm = load_json('metrics', 'pair_metrics.json', args=args)
    rt = load_json('metrics', 'retrieval_primary.json', args=args)
    gv = load_json('metrics', 'govdocs1.json', args=args)
    pr = load_json('metrics', 'png_repair.json', args=args)
    prep = load_json('metrics', 'prepare.json', args=args)
    w = sel['weights_selected']
    di = prep.get('dataset_info', {})
    parts = ['# Results', '',
             f"Dataset: `{args.data_root}` ({di.get('source', 'dataset package')}; similarities: "
             f"{di.get('features', 'recompute')}; target: {di.get('target', 'eq23')})  ",
             f"Seeds: {args.seeds}; learned methods are shown as mean ± SD over the seeds.  ",
             f"Selected temperature: {sel['tau_selected']:.2f}; w* = "
             + ', '.join(f'{config.COMPONENT_NAMES[c]} {x:.4f}' for c, x in zip(config.COMPONENTS, w)) + '.', '']

    # Table 4 ------------------------------------------------------------------------------------
    cols = ['R2', 'MSE', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'AUC', 'AP']
    order = [('Uniform fusion', pm['fixed']['Uniform fusion']), ('Weighted linear fusion', pm['fixed']['Weighted linear fusion']),
             ('2-gram regressor', pm['learned']['2-gram regressor']),
             ('Matched binary classifier', pm['learned']['Matched binary classifier']),
             ('Unweighted regressor', pm['learned']['Unweighted regressor']), ('LAWSR', pm['learned']['LAWSR'])]
    rows = [(n, {c: cell(e, c) for c in cols}) for n, e in order]
    parts.append(table(rows, cols, 'table4', args, 'TABLE 4 Test-pair performance against the provenance reference',
                       f"Binary-classifier probabilities multiplied by the mean positive training reference "
                       f"({prep['mean_positive_reference_train']:.4f}) for the graded-error metrics."))

    # Table 5 ------------------------------------------------------------------------------------
    t5 = pd.DataFrame([{'Component': config.COMPONENT_NAMES[c],
                        'Validation R2': fmt(lrn['components'][c]['R2'], 3),
                        'Validation Spearman': fmt(lrn['components'][c]['Spearman'], 3),
                        'Learnability': fmt(lrn['learnability'][c], 4), 'Weight': fmt(x, 4)}
                       for c, x in zip(config.COMPONENTS, w)])
    t5.to_csv(path(args, 'tables', 'table5.csv'), index=False)
    grid = pd.DataFrame([{'tau': f"{g['tau']:.2f}", 'mean validation MSE': f"{g['val_mse_mean']:.6f}",
                          **{config.COMPONENT_NAMES[c]: f'{x:.4f}' for c, x in zip(config.COMPONENTS, g['weights'])}}
                         for g in sel['grid']])
    grid.to_csv(path(args, 'tables', 'temperature_grid.csv'), index=False)
    parts.append('### TABLE 5 Validation estimates of component learnability and weights\n\n' + md(t5) +
                 f"\n\nLearnability from the {lrn['policy']} validation R2 and Spearman over the seeds; weights at the "
                 f"selected temperature {sel['tau_selected']:.2f}.\n\nTemperature grid (selection by mean validation MSE):\n\n"
                 + md(grid) + '\n')

    # Table 6 ------------------------------------------------------------------------------------
    names = [('Unscaled', 'Unweighted regressor', config.UNSCALED), ('Equal', 'Equal scaling', config.EQUAL_SCALES),
             ('Heuristic', 'Heuristic scaling', config.HEURISTIC_SCALES), ('Learnability-aware', 'LAWSR', w)]
    short = ['Histogram', 'Entropy', 'Byte 2-gram', 'Compression']
    rows = []
    for label, key, sc in names:
        d = {col: (f'{x:.4f}' if label == 'Learnability-aware' else f'{x:.3f}', x, None) for col, x in zip(short, sc)}
        d.update({'R2': cell(pm['learned'][key], 'R2'), 'MAE': cell(pm['learned'][key], 'MAE'),
                  'Same-file AUC': cell(pm['learned'][key], 'AUC')})
        rows.append((label, d))
    parts.append(table(rows, ['Histogram', 'Entropy', 'Byte 2-gram', 'Compression', 'R2', 'MAE', 'Same-file AUC'],
                       'table6', args, 'TABLE 6 Effect of input scaling in the matched regression network'))

    # Table 7 ------------------------------------------------------------------------------------
    cols = ['Top-1 (%)', 'Top-10 (%)', 'MRR', 'Same-file P@10 (%)', 'Within-file Spearman']
    ro = rt['random_ordering']
    rows = [('Random ordering', {'Top-1 (%)': (fmt(ro['Top-1 (%)'], 3), ro['Top-1 (%)'], None),
                                 'Top-10 (%)': (fmt(ro['Top-10 (%)'], 3), ro['Top-10 (%)'], None),
                                 'MRR': (fmt(ro['MRR'], 4), ro['MRR'], None),
                                 'Same-file P@10 (%)': (fmt(ro['Same-file P@10 (%)'], 1), ro['Same-file P@10 (%)'], None),
                                 'Within-file Spearman': ('0.000', 0.0, None)})]
    for n in ['Histogram cosine', 'Entropy similarity', 'Byte 2-gram Jaccard', 'Compression similarity', 'Uniform fusion',
              'Weighted linear fusion']:
        e = {**rt['fixed'][n], 'Within-file Spearman': pm['fixed'][n]['Within-file Spearman']}
        rows.append((n, {c: cell(e, c) for c in cols}))
    for n in ['Matched binary classifier', 'Unweighted regressor', 'LAWSR']:
        e = {'per_seed': {s: {**v, 'Within-file Spearman': pm['learned'][n]['per_seed'][s]['Within-file Spearman']}
                          for s, v in rt['learned'][n]['per_seed'].items()}}
        rows.append((n, {c: cell(e, c) for c in cols}))
    qx = ', '.join(f'{k.upper()} {v}' for k, v in rt['queries_by_extension'].items())
    parts.append(table(rows, cols, 'table7', args, 'TABLE 7 Ranking consistency and exact successor retrieval',
                       f"{rt['queries']} eligible queries; {rt['candidates_per_query']} candidates per query. "
                       f"Queries by extension: {qx}."))

    # Table 8 ------------------------------------------------------------------------------------
    labels = {'aligned_512': 'Govdocs1 aligned 512 B', 'aligned_256': 'Aligned 256 B', 'aligned_1024': 'Aligned 1,024 B',
              'aligned_4096': 'Aligned 4,096 B', 'unaligned_512': 'Unaligned 512 B', 'gzip_512': 'Gzip payload',
              'aes256ctr_512': 'AES-256-CTR payload'}
    rows = []
    for k, lab in labels.items():
        if gv.get('skipped') or k not in gv['conditions']:
            continue
        c = gv['conditions'][k]
        la = {'per_seed': {s: {'AUC': v} for s, v in c['lawsr_auc_per_seed'].items()}}
        lt = {'per_seed': c['lawsr_retrieval_per_seed']}
        rows.append((lab, {'LAWSR AUC': cell(la, 'AUC'), 'Uniform AUC': (fmt(c['uniform_auc'], 3), c['uniform_auc'], None),
                           'Single-score AUC': (fmt(c['single_score_auc'], 3), c['single_score_auc'], None),
                           'LAWSR Top-10 (%)': cell(lt, 'Top-10 (%)'),
                           'Uniform Top-10 (%)': cell(c['uniform_retrieval'], 'Top-10 (%)'),
                           'LAWSR MRR': cell(lt, 'MRR'),
                           'Random Top-10 (%)': (fmt(c['random_ordering']['Top-10 (%)'], 3), c['random_ordering']['Top-10 (%)'], None)}))
    if rows:
        parts.append(table(rows, ['LAWSR AUC', 'Uniform AUC', 'Single-score AUC', 'LAWSR Top-10 (%)', 'Uniform Top-10 (%)',
                                  'LAWSR MRR', 'Random Top-10 (%)'], 'table8', args,
                           'TABLE 8 Govdocs1 transfer and stress results with the frozen model',
                           f"Single score: {config.COMPONENT_NAMES[gv['single_score_component']]} (highest validation learnability)."))
    else:
        parts.append('### TABLE 8 Govdocs1 transfer and stress results\n\nNot computed: the Govdocs1 sets were not built '
                     '(dataset/build_govdocs1.py).\n')

    # Table 9 and Figure 3 -------------------------------------------------------------------------
    def grouped(key):
        per = pm[key]
        groups = [r['group'] for r in next(iter(per.values()))]
        out = []
        for i, g in enumerate(groups):
            vals = {k: [per[s][i][k] for s in per] for k in ['pairs', 'mean_target', 'mean_prediction', 'MAE', 'MSE', 'mean_residual']}
            out.append((g, vals))
        return out
    mode_names = {'same_file': 'Same file', 'same_extension_different_file': 'Same extension, different file',
                  'different_extension': 'Different extension', 'random': 'Random pairs', 'Aggregate': 'Aggregate'}
    t9 = []
    for panel, key, ren in [('A', 'error_by_mode', mode_names), ('B', 'error_by_band', {})]:
        rows = []
        for g, v in grouped(key):
            mp, ma, ms = mean_sd(v['mean_prediction']), mean_sd(v['MAE']), mean_sd(v['MSE'])
            rows.append({'Group': ren.get(g, g), 'Pairs': f"{int(v['pairs'][0]):,}",
                         'Mean target': fmt(np.mean(v['mean_target']), 4),
                         'Mean prediction': fmt(mp[0], 4, mp[1]), 'MAE': fmt(ma[0], 3, ma[1]), 'MSE': fmt(ms[0], 4, ms[1]),
                         'Mean residual': fmt(mean_sd(v['mean_residual'])[0], 4)})
        df = pd.DataFrame(rows)
        df.to_csv(path(args, 'tables', f'table9{panel.lower()}.csv'), index=False)
        t9.append(f"Panel {panel}. {'Pair generation modes' if panel == 'A' else 'Same-file reference bands'}.\n\n" + md(df))
    parts.append('### TABLE 9 Prediction error by pair generation mode and within-file reference band (LAWSR)\n\n'
                 + '\n\n'.join(t9) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for (g, v) in grouped('error_by_band')[:-1]:
            ax.plot(np.mean(v['mean_target']), np.mean(v['mean_prediction']), 'o', color='black')
        for (g, v) in grouped('error_by_mode')[1:4]:
            ax.plot(np.mean(v['mean_target']), np.mean(v['mean_prediction']), 's', mfc='white', mec='black')
        ax.plot([0, 1], [0, 1], '--', color='grey', lw=1)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlabel('Mean reference score')
        ax.set_ylabel('Mean predicted score')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(path(args, 'figures', f'figure3.{ext}'), dpi=300)
        parts.append('Figure 3: `figures/figure3.png` (filled circles: same-file bands; open squares: other sampling modes).\n')
    except ImportError:
        parts.append('Figure 3 not drawn (matplotlib is not installed).\n')

    # Table 10 -----------------------------------------------------------------------------------
    if pr.get('skipped'):
        parts.append('### TABLE 10 Byte-exact PNG reconstruction\n\nNot computed: the repair task needs the intact test PNG '
                     'files (dataset/prepare_kaggle.py --png-sources).\n')
    else:
        rows = []
        for n in ['Uniform fusion', 'Direct byte 2-gram Jaccard', 'Unweighted regressor', 'LAWSR']:
            v = pr['summary'][n]
            rows.append((n, {k: (str(v[k]) if k != 'Exact recovery (%)' else f'{v[k]:.1f}', v[k], None)
                             for k in ['Files', 'Top-10 hits', 'Exact', 'Wrong', 'Abstained', 'Exact recovery (%)']}))
        parts.append(table(rows, ['Files', 'Top-10 hits', 'Exact', 'Wrong', 'Abstained', 'Exact recovery (%)'], 'table10', args,
                           'TABLE 10 Byte-exact PNG reconstruction after candidate ranking and validation',
                           f"Learned methods use the seed-{pr['checkpoint_seed']} checkpoint; the validator checks at most "
                           f"{pr['max_checks']} candidates."))

    # Table 11 -----------------------------------------------------------------------------------
    rows = []
    for n, key in [('All four content components', 'LAWSR')] + [
            (f'Without {config.COMPONENT_NAMES[c].lower()}', f'Without {config.COMPONENT_NAMES[c].lower()}') for c in config.COMPONENTS]:
        e, r = pm['learned'][key], rt['learned'][key]
        rows.append((n, {'R2': cell(e, 'R2'), 'RMSE': cell(e, 'RMSE'), 'MAE': cell(e, 'MAE'), 'Same-file AUC': cell(e, 'AUC'),
                         'Top-10 (%)': cell({'per_seed': r['per_seed']}, 'Top-10 (%)')}))
    parts.append(table(rows, ['R2', 'RMSE', 'MAE', 'Same-file AUC', 'Top-10 (%)'], 'table11a', args,
                       'TABLE 11 Panel A. Content ablation (remaining scales renormalized to sum to one, refitted)'))
    rows = [('Four content inputs', {'Corrupted labels (%)': ('0', 0, None), 'R2': cell(pm['learned']['LAWSR'], 'R2'),
                                     'MAE': cell(pm['learned']['LAWSR'], 'MAE'), 'Same-file AUC': cell(pm['learned']['LAWSR'], 'AUC')}),
            ('Content plus correct type cue', {'Corrupted labels (%)': ('0', 0, None),
                                               'R2': cell(pm['learned']['Content plus correct type cue'], 'R2'),
                                               'MAE': cell(pm['learned']['Content plus correct type cue'], 'MAE'),
                                               'Same-file AUC': cell(pm['learned']['Content plus correct type cue'], 'AUC')})]
    for pct, v in pm['type_corruption'].items():
        rows.append(('Type-assisted model', {'Corrupted labels (%)': (f"{pct} ({v['files']} files, {v['percent_of_test_files']:.2f}%)", float(pct), None),
                                             'R2': cell(v, 'R2'), 'MAE': cell(v, 'MAE'), 'Same-file AUC': cell(v, 'AUC')}))
    parts.append(table(rows, ['Corrupted labels (%)', 'R2', 'MAE', 'Same-file AUC'], 'table11b', args,
                       'TABLE 11 Panel B. Sensitivity to optional file-extension labels'))

    with open(path(args, 'RESULTS.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts) + '\n')
    print(f"wrote {path(args, 'RESULTS.md')}")


if __name__ == '__main__':
    main()
