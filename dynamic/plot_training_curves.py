"""
Training Curves from PTB Log — TensorBoard-style visualization
Uses data from logs/progress_log_ptb.txt
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("evaluation_charts", exist_ok=True)

# ============================================================================
# PARSE PTB LOG
# ============================================================================
log_text = open("logs/progress_log_ptb.txt").read()

def parse_model(text, model_name_pattern):
    """Extract (step, val_bpc) pairs for a model."""
    pattern = rf"\[{re.escape(model_name_pattern)}\] (\d+)/40000 \| Val BPC=([0-9.]+) \| Val NLL"
    matches = re.findall(pattern, text)
    return [(int(s), float(b)) for s, b in matches]

models_raw = {
    'Baseline LSTM': parse_model(log_text, 'LSTM_Baseline_1000'),
    'LN Baseline':   parse_model(log_text, 'LayerNorm_LSTM_1000'),
    'HyperLSTM':     parse_model(log_text, 'HyperLSTM_1000'),
    'LN HyperLSTM':  parse_model(log_text, 'LayerNorm_HyperLSTM_1000'),
}

colors = {
    'Baseline LSTM': '#E74C3C',   # Red
    'LN Baseline':   '#27AE60',   # Green
    'HyperLSTM':     '#E67E22',   # Orange
    'LN HyperLSTM':  '#2980B9',   # Blue
}

test_bpc = {
    'Baseline LSTM': 1.3623,
    'LN Baseline':   1.3564,
    'HyperLSTM':     1.3057,
    'LN HyperLSTM':  1.2837,
}

# ============================================================================
# CHART STYLE
# ============================================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================================
# 10. TRAINING CURVES — Val BPC vs Steps
# ============================================================================
def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, data in models_raw.items():
        steps = [d[0] for d in data]
        bpc = [d[1] for d in data]
        ax.plot(steps, bpc, label=name, color=colors[name], linewidth=1.8, alpha=0.9)

    ax.set_xlabel('Training Steps (x1000)', fontsize=11)
    ax.set_ylabel('Validation BPC', fontsize=11)
    ax.set_title('PTB — Validation BPC vs Training Steps\n(H=1000, 40K steps)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 40000)
    ax.set_ylim(1.25, 2.0)

    # Mark test BPC at end
    for name, data in models_raw.items():
        final_step = data[-1][0]
        final_bpc = data[-1][1]
        ax.scatter([final_step], [final_bpc], color=colors[name], s=40, zorder=5)
        ax.annotate(f'{test_bpc[name]:.4f}',
                    xy=(final_step, test_bpc[name]),
                    xytext=(final_step + 800, test_bpc[name] + 0.05),
                    fontsize=7.5, color=colors[name], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colors[name], lw=0.8))

    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xticklabels(['0', '10K', '20K', '30K', '40K'])

    # Right: Zoom on convergence
    ax2 = axes[1]
    for name, data in models_raw.items():
        steps = [d[0] for d in data]
        bpc = [d[1] for d in data]
        ax2.plot(steps, bpc, label=name, color=colors[name], linewidth=1.8, alpha=0.9)

    ax2.set_xlabel('Training Steps (x1000)', fontsize=11)
    ax2.set_ylabel('Validation BPC', fontsize=11)
    ax2.set_title('PTB — Convergence Detail (steps 5K–40K)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_xlim(5000, 40000)
    ax2.set_ylim(1.25, 1.55)

    ax2.set_xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
    ax2.set_xticklabels(['5K', '10K', '15K', '20K', '25K', '30K', '35K', '40K'])

    plt.suptitle('Training Curves — PTB Character-Level LM (Section 5)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('evaluation_charts/10_training_curves.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 10_training_curves.png")


# ============================================================================
# 11. PAPER-STYLE COMPARISON
# ============================================================================
def plot_paper_style():
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in models_raw.items():
        steps = [d[0] for d in data]
        bpc = [d[1] for d in data]
        ax.plot(steps, bpc, label=name, color=colors[name], linewidth=2.0, alpha=0.9)

    # Paper reference lines (Table 3, H=1800)
    paper_bpc = {
        'Baseline LSTM': 1.312,
        'LN Baseline':   1.267,
        'HyperLSTM':     1.265,
        'LN HyperLSTM':  1.250,
    }
    for name, pval in paper_bpc.items():
        ax.axhline(y=pval, color=colors[name], linestyle='--', linewidth=1.2,
                   alpha=0.5, label=f'{name} (paper)' if 'paper' not in name else None)

    ax.set_xlabel('Training Steps (x1000)', fontsize=12)
    ax.set_ylabel('Validation BPC', fontsize=12)
    ax.set_title('PTB Test BPC: Our Implementation vs Paper Reference\n'
                 '(Solid=ours H=1000, Dashed=paper H=1800, 400K steps)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 40000)
    ax.set_ylim(1.1, 2.0)

    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xticklabels(['0', '10K', '20K', '30K', '40K'])

    plt.tight_layout()
    plt.savefig('evaluation_charts/11_paper_comparison_curves.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 11_paper_comparison_curves.png")


# ============================================================================
# 12. FINAL COMPARISON TABLE + BAR
# ============================================================================
def plot_final_comparison():
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                              gridspec_kw={'width_ratios': [1.2, 1]})

    names = ['Baseline LSTM', 'LN Baseline', 'HyperLSTM', 'LN HyperLSTM']
    val_bpc = [1.3931, 1.3897, 1.3802, 1.3697]
    test_bpc_vals = [1.3623, 1.3564, 1.3057, 1.2837]
    clrs = [colors[n] for n in names]

    # Table
    ax_table.axis('off')
    col_labels = ['Model', 'Val BPC', 'Test BPC', 'Gap']
    col_data = []
    for n, v, t in zip(names, val_bpc, test_bpc_vals):
        gap = v - t
        col_data.append([n, f'{v:.4f}', f'{t:.4f}', f'{gap:+.4f}'])

    table = ax_table.table(cellText=col_data, colLabels=col_labels,
                             loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            bg = '#F8F9FA' if row % 2 == 1 else 'white'
            cell.set_facecolor(bg)
            if row == 4:
                cell.set_facecolor('#D5F4E6')
                cell.set_text_props(color='#1E8449', fontweight='bold')

    ax_table.set_title('Final Results — PTB (H=1000, 40K steps)',
                       fontsize=13, fontweight='bold', pad=20)

    # Bar chart
    x = np.arange(len(names))
    bars = ax_bar.bar(x, test_bpc_vals, color=clrs, width=0.6,
                       edgecolor='white', linewidth=1.5)
    bars[-1].set_edgecolor('#27AE60')
    bars[-1].set_linewidth(2.5)

    for bar, val, name in zip(bars, test_bpc_vals, names):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        imp = (test_bpc_vals[0] - val) / test_bpc_vals[0] * 100
        if imp > 0:
            ax_bar.text(bar.get_x() + bar.get_width()/2, val - 0.025,
                        f'-{imp:.1f}%', ha='center', va='top',
                        fontsize=8, color='green', fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['Baseline', 'LN', 'Hyper', 'LN+Hyper'],
                            fontsize=10, rotation=15)
    ax_bar.set_ylabel('Test BPC (lower = better)', fontsize=11)
    ax_bar.set_title('Test BPC Improvement', fontsize=13, fontweight='bold')
    ax_bar.set_ylim(1.2, 1.45)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax_bar.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('evaluation_charts/12_final_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 12_final_comparison.png")


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    print("Generating training curve charts...")
    print("=" * 60)
    plot_training_curves()
    plot_paper_style()
    plot_final_comparison()
    print("=" * 60)
    print("Done! Charts saved in ./evaluation_charts/")
