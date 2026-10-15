"""Plot per-origin gradient similarity to the last origin from exp_grad.py json. usage: python plot_grad.py in.json out.png"""
import sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); P = 48
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
sty = {'init': ('tab:gray', 'init'), 'A1trained': ('tab:blue', 'A1 trained'), 'E1trained': ('tab:orange', 'E1 trained (last-only)')}
for ax, g in zip(axs.flat, ('all', 'G', 'mixer', 'embed')):
    for r in R:
        c, lab = sty.get(r['name'], ('k', r['name'])); K = len(r[g]['cos_mean_to_last']); x = list(range(K))
        ax.plot(x, r[g]['cos_mean_to_last'], '-o', color=c, ms=4, label=f'{lab}: cos(E g_j, E g_last)')
        ax.plot(x, r[g]['batch_cos_to_last'], '--s', color=c, ms=3, alpha=.7, label=f'{lab}: E cos(g_j, g_last) per batch')
        ax.plot(x, r[g]['split_half_cos'], ':', color=c, alpha=.6, label=f'{lab}: split-half cos (reliability of E g_j)')
    ax.axhline(0, color='k', lw=.6); ax.set_title(f'params: {g}'); ax.set_ylim(-1.05, 1.05); ax.grid(alpha=.3)
    ax.set_xticks(x); ax.set_xticklabels([f'{j}\n{(j+1)*P}h' for j in x], fontsize=7)
for ax in axs[1]: ax.set_xlabel('patch origin j  /  history length')
for ax in axs[:, 0]: ax.set_ylabel('cosine similarity to last-origin gradient')
axs[0, 0].legend(fontsize=7, loc='lower right')
fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else "PAT-304 A1: per-origin gradient vs last-origin gradient")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130); print('saved', sys.argv[2])
