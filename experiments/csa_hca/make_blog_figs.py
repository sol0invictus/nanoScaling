"""Generate two standalone blog figures from long_context_results.json:
  ppl_vs_length.png        — perplexity vs context length (both models/protocols)
  per_position_4096.png    — per-token NLL at L=4096 (coherent docs, both models)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = os.path.join(ROOT, 'experiments/csa_hca/long_context_results.json')
OUT = os.path.join(ROOT, 'experiments/csa_hca')
res = json.load(open(J))
b, c = res['models']['baseline'], res['models']['csa']
BLUE, RED = '#1f5fb4', '#d12f2f'

def lens(proto):  # JSON keys are strings; return sorted int lengths present for both models
    L = sorted(int(k) for k in b[proto] if k in c[proto])
    return L

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

# ---- Figure 1: perplexity vs context length -------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for proto, style, lab in [('stream', '-o', 'concatenated stream'),
                          ('longdoc', '--s', 'coherent long docs')]:
    L = lens(proto)
    ax.plot(L, [b[proto][str(x)]['ppl'] for x in L], style, color=BLUE, lw=2, ms=7,
            label=f'baseline · {lab}')
    ax.plot(L, [c[proto][str(x)]['ppl'] for x in L], style, color=RED, lw=2, ms=7,
            label=f'CSA · {lab}')
ax.axvline(1024, color='gray', ls=':', lw=1.2)
ax.text(1024, ax.get_ylim()[1]*0.99, ' training length', color='gray', va='top', fontsize=10)
allL = lens('stream')
ax.set_xticks(allL); ax.set_xticklabels([f'{x}\n({x//1024 if x>=1024 else 0.5}×)' for x in allL])
ax.set_xlabel('context length (tokens)'); ax.set_ylabel('perplexity (↓ better)')
ax.set_title('Zero-shot perplexity vs. context length')
ax.legend(fontsize=10, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'ppl_vs_length.png')); plt.close(fig)

# ---- Figure 2: per-position NLL at L=4096 ---------------------------------- #
def smooth(v, k=41):
    v = np.asarray(v, float); ker = np.ones(k)/k
    return np.convolve(v, ker, mode='valid')
src = 'longdoc' if '4096' in b.get('longdoc', {}) else 'stream'
pb, pc = b[src]['4096']['pos_nll'], c[src]['4096']['pos_nll']
off = 20
xs = np.arange(len(pb))[off:-off]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(xs, smooth(pb), color=BLUE, lw=2, label='baseline (standard attention)')
ax.plot(xs, smooth(pc), color=RED, lw=2, label='CSA')
ax.axvline(1024, color='gray', ls=':', lw=1.2)
ax.text(1044, ax.get_ylim()[1]*0.995, 'training length', color='gray', va='top', ha='left', fontsize=10)
ax.set_xlabel('token position in sequence'); ax.set_ylabel('NLL (nats, smoothed)  (↓ better)')
ax.set_title('Per-token loss at 4096-token context')
ax.legend(fontsize=11, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'per_position_4096.png')); plt.close(fig)

print("wrote:", os.path.join(OUT, 'ppl_vs_length.png'))
print("wrote:", os.path.join(OUT, 'per_position_4096.png'))
print(f"per-position source protocol: {src}, docs/window n={b[src]['4096']['n_windows']}")
