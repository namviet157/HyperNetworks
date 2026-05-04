"""
Evaluation Visualization for Dynamic HyperNetwork Presentation
Creates professional charts and tables for Section 5: Evaluation (20%)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("evaluation_charts", exist_ok=True)

# ============================================================================
# CHART STYLE SETUP
# ============================================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#E67E22',      # Orange
    'hyper': '#2980B9',          # Blue
    'ln': '#27AE60',             # Green
    'ln_hyper': '#8E44AD',       # Purple
    'light': '#F39C12',          # Light orange
    'dark_blue': '#1A5276',      # Dark blue
    'gray': '#7F8C8D',
    'red': '#E74C3C',
    'success': '#2ECC71',
}

# ============================================================================
# 1. SUMMARY TABLE — All 3 datasets
# ============================================================================
def plot_summary_table():
    """Create a summary table image for all 3 datasets"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    # Data
    headers = ['Model', 'PTB\nTest BPC', 'enwik8\nTest BPC', 'Shakespeare\nTest BPC', 'Speed']
    data = [
        ['Baseline LSTM', '1.3623', '1.8568', '2.0619', '~7.5 it/s'],
        ['LN Baseline',  '1.3564', '1.6913', '1.9889', '~5.4 it/s'],
        ['HyperLSTM',     '1.3057', '1.8102', '2.0261', '~3.4 it/s'],
        ['LN HyperLSTM',  '1.2837', '1.6439', '1.9438', '~3.4 it/s'],
    ]

    col_widths = [0.22, 0.18, 0.18, 0.20, 0.18]
    row_height = 0.18
    start_y = 0.78

    # Header
    x = 0.02
    for i, (h, w) in enumerate(zip(headers, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, start_y), w, row_height,
            boxstyle="round,pad=0.01", facecolor='#2C3E50',
            edgecolor='none', transform=ax.transAxes, clip_on=False))
        ax.text(x + w/2, start_y + row_height/2, h,
                ha='center', va='center', color='white',
                fontweight='bold', transform=ax.transAxes)
        x += w + 0.008

    # Data rows
    for row_idx, row in enumerate(data):
        y = start_y - (row_idx + 1) * (row_height + 0.01)
        x = 0.02
        bg_color = '#F8F9FA' if row_idx % 2 == 0 else 'white'

        for col_idx, (val, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w, row_height,
                boxstyle="round,pad=0.005", facecolor=bg_color,
                edgecolor='#E0E0E0', linewidth=0.5,
                transform=ax.transAxes, clip_on=False))

            # Bold model name
            fontweight = 'bold' if col_idx == 0 else 'normal'
            color = 'black'

            # Highlight best values
            if row_idx == 3 and col_idx in [1, 2, 3]:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x, y), w, row_height,
                    boxstyle="round,pad=0.005", facecolor='#D5F4E6',
                    edgecolor='#27AE60', linewidth=1,
                    transform=ax.transAxes, clip_on=False))
                color = '#1E8449'
                fontweight = 'bold'

            ax.text(x + w/2, y + row_height/2, val,
                    ha='center', va='center', color=color,
                    fontweight=fontweight, transform=ax.transAxes)
            x += w + 0.008

    ax.set_title('Bảng tổng hợp kết quả — Test BPC (thấp hơn = tốt hơn)',
                 fontsize=14, fontweight='bold', pad=20, loc='center')
    plt.tight_layout()
    plt.savefig('evaluation_charts/01_summary_table.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 01_summary_table.png")


# ============================================================================
# 2. BAR CHART — PTB Test BPC Comparison
# ============================================================================
def plot_ptb_barchart():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['Baseline\nLSTM', 'LN\nBaseline', 'HyperLSTM', 'LN\nHyperLSTM']
    values = [1.3623, 1.3564, 1.3057, 1.2837]
    colors = [COLORS['baseline'], COLORS['ln'], COLORS['hyper'], COLORS['ln_hyper']]

    bars = ax.bar(models, values, color=colors, width=0.55,
                  edgecolor='white', linewidth=1.5)

    # Highlight best
    bars[3].set_edgecolor('#27AE60')
    bars[3].set_linewidth(2.5)

    # Values on bars
    for bar, val in zip(bars, values):
        ypos = val + 0.008
        improvement = (values[0] - val) / values[0] * 100
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val}\n(-{improvement:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Test BPC (bits per character)', fontsize=12)
    ax.set_title('PTB Dataset — Test BPC Comparison\n(HyperLSTM vs Baseline)', fontsize=13, fontweight='bold')
    ax.set_ylim(1.22, 1.42)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, ['Baseline', 'LN Baseline', 'HyperLSTM', 'LN HyperLSTM'])]
    ax.legend(handles=patches, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('evaluation_charts/02_ptb_barchart.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 02_ptb_barchart.png")


# ============================================================================
# 3. GROUPED BAR — All 3 datasets
# ============================================================================
def plot_all_datasets_grouped():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    datasets = ['PTB', 'enwik8', 'Shakespeare']
    x = np.arange(len(datasets))
    width = 0.18

    models_data = {
        'Baseline LSTM':  [1.3623, 1.8568, 2.0619],
        'LN Baseline':    [1.3564, 1.6913, 1.9889],
        'HyperLSTM':       [1.3057, 1.8102, 2.0261],
        'LN HyperLSTM':    [1.2837, 1.6439, 1.9438],
    }
    colors = [COLORS['baseline'], COLORS['ln'], COLORS['hyper'], COLORS['ln_hyper']]

    for i, (name, vals) in enumerate(models_data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i],
                      edgecolor='white', linewidth=1)
        # Labels on top
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold' if i == 3 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel('Test BPC (lower = better)', fontsize=12)
    ax.set_title('Test BPC across All 3 Datasets', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(1.1, 2.3)

    # Annotation for best (LN HyperLSTM)
    for i, val in enumerate(models_data['LN HyperLSTM']):
        ax.annotate('★ Best', xy=(x[i] + 1.5*width, val),
                    xytext=(x[i] + 1.5*width, val + 0.06),
                    ha='center', fontsize=8, color='#27AE60', fontweight='bold')

    plt.tight_layout()
    plt.savefig('evaluation_charts/03_all_datasets_grouped.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 03_all_datasets_grouped.png")


# ============================================================================
# 4. IMPROVEMENT % BAR CHART
# ============================================================================
def plot_improvement_bars():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['PTB', 'enwik8', 'Shakespeare']
    hyper_imp = []  # HyperLSTM vs Baseline
    ln_hyper_imp = []  # LN HyperLSTM vs Baseline

    for ds, baseline, hyper, ln_hyper in [
        ('PTB', 1.3623, 1.3057, 1.2837),
        ('enwik8', 1.8568, 1.8102, 1.6439),
        ('Shakespeare', 2.0619, 2.0261, 1.9438),
    ]:
        hyper_imp.append((baseline - hyper) / baseline * 100)
        ln_hyper_imp.append((baseline - ln_hyper) / baseline * 100)

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, hyper_imp, width, label='HyperLSTM vs Baseline',
                   color=COLORS['hyper'], edgecolor='white')
    bars2 = ax.bar(x + width/2, ln_hyper_imp, width, label='LN HyperLSTM vs Baseline',
                   color=COLORS['ln_hyper'], edgecolor='white')

    for bar, val in zip(bars1, hyper_imp):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, ln_hyper_imp):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('HyperLSTM Improvement over Baseline\n(lower BPC = higher improvement)',
                 fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig('evaluation_charts/04_improvement_bars.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 04_improvement_bars.png")


# ============================================================================
# 5. ABLATION RESULTS BAR CHART
# ============================================================================
def plot_ablation_results():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Experiment data
    experiments = [
        {
            'title': 'Exp1: Recurrent Dropout',
            'xlabel': 'Variant',
            'labels': ['Baseline\n(no dropout)', 'HyperLSTM'],
            'values': [1.5987, 1.5969],
            'colors': [COLORS['baseline'], COLORS['hyper']],
            'best_idx': 1,
            'note': 'Dropout helps regularization'
        },
        {
            'title': 'Exp2: LN Position',
            'xlabel': 'Position',
            'labels': ['LN_before\n(paper)', 'LN_after\n(wrong)'],
            'values': [1.5954, 5.8690],
            'colors': [COLORS['success'], COLORS['red']],
            'best_idx': 0,
            'note': 'LN before activation is CRITICAL'
        },
        {
            'title': 'Exp3: Weight Init',
            'xlabel': 'Init method',
            'labels': ['Orthogonal', 'Xavier'],
            'values': [1.5877, 1.5647],
            'colors': [COLORS['baseline'], COLORS['hyper']],
            'best_idx': 1,
            'note': 'Xavier slightly better in 20K steps'
        },
        {
            'title': 'Exp4: Nz (bottleneck)',
            'xlabel': 'Nz value',
            'labels': ['Nz=1', 'Nz=4\n(paper)', 'Nz=16'],
            'values': [1.5555, 1.5978, 1.6008],
            'colors': [COLORS['success'], COLORS['baseline'], COLORS['gray']],
            'best_idx': 0,
            'note': 'Smaller Nz = stronger regularization'
        },
        {
            'title': 'Exp5: Regularization',
            'xlabel': 'Method',
            'labels': ['Orthogonal', 'Normal_small'],
            'values': [1.5771, 1.6052],
            'colors': [COLORS['baseline'], COLORS['gray']],
            'best_idx': 0,
            'note': 'Orthogonal init gives stability'
        },
    ]

    for ax, exp in zip(axes, experiments):
        bars = ax.bar(exp['labels'], exp['values'], color=exp['colors'],
                     width=0.55, edgecolor='white', linewidth=1.5)
        bars[exp['best_idx']].set_edgecolor('#27AE60')
        bars[exp['best_idx']].set_linewidth(2)

        for bar, val in zip(bars, exp['values']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.05 if val < 3 else 0.15),
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    color='green' if bar == bars[exp['best_idx']] else 'black')

        ax.set_title(exp['title'], fontsize=11, fontweight='bold')
        ax.set_xlabel(exp['xlabel'], fontsize=10)
        ax.set_ylabel('Val BPC', fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        # Set ylim for LN_after experiment (very high value)
        if exp['title'] == 'Exp2: LN Position':
            ax.set_ylim(0, 6.5)
        else:
            ax.set_ylim(1.4, 1.7)

        ax.text(0.5, -0.18, exp['note'], ha='center', fontsize=8.5,
                 style='italic', color='#555', transform=ax.transAxes)

    plt.suptitle('Ablation Study Results on PTB (Val BPC — lower = better)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('evaluation_charts/05_ablation_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 05_ablation_results.png")


# ============================================================================
# 6. TRAINING SPEED COMPARISON
# ============================================================================
def plot_speed_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['Baseline\nLSTM', 'LN\nBaseline', 'HyperLSTM', 'LN\nHyperLSTM']
    speeds = [7.5, 5.4, 3.4, 3.4]  # it/s approximate
    colors = [COLORS['baseline'], COLORS['ln'], COLORS['hyper'], COLORS['ln_hyper']]

    bars = ax.bar(models, speeds, color=colors, width=0.55,
                  edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f'{val} it/s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Training Speed (iterations/sec)', fontsize=12)
    ax.set_title('Training Speed Comparison\n(HyperLSTM ~2x slower than Baseline)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Annotation
    ax.annotate('HyperLSTM is ~2x slower\ndue to extra hyper cell',
                xy=(2.5, 3.4), xytext=(3.2, 6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray']),
                fontsize=9, color=COLORS['gray'])

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, models)]
    ax.legend(handles=patches, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('evaluation_charts/06_speed_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 06_speed_comparison.png")


# ============================================================================
# 7. PARAMETER COUNT COMPARISON
# ============================================================================
def plot_params_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Parameter count
    configs = [
        ('Baseline\nLSTM',  'PTB/enwik8'),
        ('HyperLSTM\n(HH=128)', 'enwik8'),
        ('HyperLSTM\n(HH=32)',  'PTB'),
        ('HyperLSTM\n(HH=16)',  'Shakespeare'),
    ]

    # Approximate parameter counts
    baseline_params = 9_000_000   # ~9M for H=1000, embed=64
    hyper_overhead = {
        'enwik8': 3_200_000,   # HH=128, HE=64
        'PTB': 500_000,          # HH=32, HE=4
        'Shakespeare': 200_000,  # HH=16, HE=4
    }

    labels = ['Baseline\nLSTM', 'HyperLSTM\n(enwik8)', 'HyperLSTM\n(PTB)', 'HyperLSTM\n(Shakespeare)']
    # Total params = baseline + hyper cell overhead
    total_params = [
        baseline_params / 1e6,
        (baseline_params + hyper_overhead['enwik8']) / 1e6,
        (baseline_params + hyper_overhead['PTB']) / 1e6,
        (baseline_params + hyper_overhead['Shakespeare']) / 1e6,
    ]
    hyper_overhead_vals = [0,
                           hyper_overhead['enwik8'] / 1e6,
                           hyper_overhead['PTB'] / 1e6,
                           hyper_overhead['Shakespeare'] / 1e6]

    x = np.arange(len(labels))
    width = 0.5
    ax1.bar(x, baseline_params / 1e6, width, label='Baseline params',
            color=COLORS['baseline'])
    ax1.bar(x, hyper_overhead_vals, width, bottom=baseline_params/1e6,
            label='Hyper cell overhead', color=COLORS['hyper'], alpha=0.8)

    for i, (base, overhead, total) in enumerate(zip(
            [baseline_params/1e6]*4, hyper_overhead_vals, total_params)):
        ax1.text(i, total + 0.2, f'{total:.1f}M', ha='center',
                 va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Total Parameters (millions)', fontsize=11)
    ax1.set_title('Parameter Count Breakdown', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)

    # Right: Overhead %
    overhead_pct = [0, 36, 6, 2]  # Approximate
    colors_bar = [COLORS['baseline'], COLORS['hyper'], COLORS['hyper'], COLORS['hyper']]
    ax2.bar(labels, overhead_pct, color=colors_bar, width=0.55,
            edgecolor='white', linewidth=1.5)
    for i, val in enumerate(overhead_pct):
        ax2.text(i, val + 0.5, f'{val}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax2.set_ylabel('Hyper Cell Overhead (%)', fontsize=11)
    ax2.set_title('Hyper Cell Parameter Overhead', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 50)

    plt.suptitle('HyperLSTM Parameter Analysis\n(H=1000, embed=64)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('evaluation_charts/07_params_overhead.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 07_params_overhead.png")


# ============================================================================
# 8. PAPER COMPARISON BAR
# ============================================================================
def plot_paper_comparison():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # PTB comparison: paper vs our impl
    labels = [
        'LSTM\n(paper)', 'LSTM\n(ours)',
        'LN LSTM\n(paper)', 'LN LSTM\n(ours)',
        'HyperLSTM\n(paper)', 'HyperLSTM\n(ours)',
        'LN HyperLSTM\n(paper)', 'LN HyperLSTM\n(ours)',
    ]
    bpc_paper = [1.312, 1.3623, 1.267, 1.3564, 1.265, 1.3057, 1.250, 1.2837]
    colors_bar = ['#F39C12', COLORS['baseline'],
                  '#27AE60', COLORS['ln'],
                  '#E67E22', COLORS['hyper'],
                  '#8E44AD', COLORS['ln_hyper']]

    x = np.arange(len(labels))
    bars = ax.bar(x, bpc_paper, color=colors_bar, width=0.65,
                  edgecolor='white', linewidth=1.2)

    for bar, val, label in zip(bars, bpc_paper, labels):
        color = '#27AE60' if 'paper' in label else 'black'
        fw = 'bold' if 'paper' in label else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=9, color=color, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Test BPC (bits per character)', fontsize=12)
    ax.set_title('PTB Results: Paper Reference vs Our Implementation (H=1000, 40K steps)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(1.15, 1.45)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#555', label='Paper reference (H=1800, 400K steps)'),
        mpatches.Patch(color='#AAA', label='Our implementation (H=1000, 40K steps)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Gap annotation
    ax.annotate('Gap due to\nsmaller H & steps',
                xy=(1, 1.3623), xytext=(1.5, 1.38),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray']),
                fontsize=9, color=COLORS['gray'], ha='center')

    plt.tight_layout()
    plt.savefig('evaluation_charts/08_paper_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 08_paper_comparison.png")


# ============================================================================
# 9. FULL EVALUATION DASHBOARD
# ============================================================================
def plot_dashboard():
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_ptb = fig.add_subplot(gs[1, 0])
    ax_enwik = fig.add_subplot(gs[1, 1])
    ax_shk = fig.add_subplot(gs[1, 2])
    ax_ablation = fig.add_subplot(gs[2, :])

    # --- Summary: grouped bar all datasets ---
    datasets = ['PTB', 'enwik8', 'Shakespeare']
    x = np.arange(len(datasets))
    width = 0.18
    models_data = {
        'Baseline':  [1.3623, 1.8568, 2.0619],
        'LN Baseline': [1.3564, 1.6913, 1.9889],
        'HyperLSTM':  [1.3057, 1.8102, 2.0261],
        'LN HyperLSTM': [1.2837, 1.6439, 1.9438],
    }
    colors = [COLORS['baseline'], COLORS['ln'], COLORS['hyper'], COLORS['ln_hyper']]
    for i, (name, vals) in enumerate(models_data.items()):
        offset = (i - 1.5) * width
        bars = ax_summary.bar(x + offset, vals, width, label=name, color=colors[i],
                              edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if i == 3:
                ax_summary.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                               fontweight='bold', color='#27AE60')
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(datasets)
    ax_summary.set_ylabel('Test BPC')
    ax_summary.set_title('Test BPC — All Datasets (lower = better)', fontweight='bold')
    ax_summary.legend(loc='upper right', fontsize=9)
    ax_summary.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_summary.set_ylim(1.1, 2.3)

    # --- Per-dataset bars ---
    for ax, ds_name, vals, model_names in [
        (ax_ptb, 'PTB', [1.3623, 1.3564, 1.3057, 1.2837],
         ['Baseline', 'LN', 'Hyper', 'LN+Hyper']),
        (ax_enwik, 'enwik8', [1.8568, 1.6913, 1.8102, 1.6439],
         ['Baseline', 'LN', 'Hyper', 'LN+Hyper']),
        (ax_shk, 'Shakespeare', [2.0619, 1.9889, 2.0261, 1.9438],
         ['Baseline', 'LN', 'Hyper', 'LN+Hyper']),
    ]:
        bars = ax.bar(model_names, vals, color=colors, width=0.6,
                      edgecolor='white', linewidth=1)
        bars[-1].set_edgecolor('#27AE60')
        bars[-1].set_linewidth(2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        ax.set_title(ds_name, fontweight='bold')
        ax.set_ylabel('Test BPC', fontsize=9)
        ax.set_xticklabels(model_names, fontsize=8, rotation=15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(min(vals) - 0.05, max(vals) + 0.1)

    # --- Ablation summary (simplified) ---
    abla_labels = ['Dropout', 'LN_before', 'LN_after', 'Orth', 'Xavier', 'Nz=1', 'Nz=4', 'Nz=16']
    abla_vals = [1.5987, 1.5954, 5.8690, 1.5877, 1.5647, 1.5555, 1.5978, 1.6008]
    abla_colors = [COLORS['baseline'], COLORS['success'], COLORS['red'],
                   COLORS['baseline'], COLORS['hyper'],
                   COLORS['success'], COLORS['baseline'], COLORS['gray']]
    best_idx = abla_vals.index(min(abla_vals))

    bars = ax_ablation.bar(abla_labels, abla_vals, color=abla_colors,
                           width=0.7, edgecolor='white', linewidth=0.8)
    bars[best_idx].set_edgecolor('#27AE60')
    bars[best_idx].set_linewidth(2)
    for bar, val in zip(bars, abla_vals):
        ax_ablation.text(bar.get_x() + bar.get_width()/2,
                         val + 0.05, f'{val:.3f}', ha='center',
                         va='bottom', fontsize=7.5, fontweight='bold',
                         color='#27AE60' if val == min(abla_vals) else 'black')

    ax_ablation.set_title('Ablation Study — Val BPC (PTB, 20K steps)',
                          fontweight='bold')
    ax_ablation.set_ylabel('Val BPC', fontsize=10)
    ax_ablation.set_xticklabels(abla_labels, fontsize=8, rotation=20)
    ax_ablation.yaxis.grid(True, linestyle='--', alpha=0.3)
    if max(abla_vals) > 3:
        ax_ablation.set_ylim(1.4, 6.5)
    else:
        ax_ablation.set_ylim(1.4, 1.8)
    ax_ablation.set_ylim(1.4, 6.5)

    plt.suptitle('Dynamic HyperNetwork — Evaluation Dashboard (Section 5)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig('evaluation_charts/09_evaluation_dashboard.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 09_evaluation_dashboard.png")


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    print("Generating evaluation charts for HyperLSTM presentation...")
    print("=" * 60)

    plot_summary_table()
    plot_ptb_barchart()
    plot_all_datasets_grouped()
    plot_improvement_bars()
    plot_ablation_results()
    plot_speed_comparison()
    plot_params_comparison()
    plot_paper_comparison()
    plot_dashboard()

    print("=" * 60)
    print(f"Done! All charts saved in ./evaluation_charts/")
    print("  Run 'open evaluation_charts/' to preview")