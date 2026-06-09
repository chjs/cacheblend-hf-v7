"""Plot F1 vs KV-cache-ratio for compblend across recompute ratios.

Reads one or more results JSONs (from blend_compress_kvzip.py) and draws, per model:
  X = KV cache ratio (1.0 .. 0.1, high→low left→right)
  Y = token-F1
  solid lines = compblend at each recompute ratio (e.g. 0.2 / 0.15 / 0.10)
  dotted = full_reuse_kvzip (rc=0 floor); dashed horizontal = full_prefill_all (ceiling)

Usage: python plot_grid.py out.png results_a.json [results_b.json ...]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_png = sys.argv[1]
    paths = sys.argv[2:]
    data = [json.load(open(p)) for p in paths]

    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 4.8), squeeze=False)
    for ax, d in zip(axes[0], data):
        kv = list(d["kvzip_ratios"])           # e.g. [1.0,0.9,...,0.1]
        rcs = list(d["recomp_ratios"])
        m = d["means"]
        ceil = m["full_prefill_all"]
        reuse = [m[f"full_reuse_kvzip@kv{r}"] for r in kv]

        ax.axhline(ceil, ls="--", color="black", lw=1.4,
                   label=f"full prefill (ceiling) = {ceil:.3f}")
        ax.plot(kv, reuse, ls=":", marker="s", ms=5, color="dimgray", lw=1.6,
                label="full reuse (rc=0, no recompute)")
        for rr in rcs:
            ys = [m[f"compblend@kv{r}_rc{rr}"] for r in kv]
            ax.plot(kv, ys, marker="o", ms=4, lw=1.8, label=f"compblend rc={rr}")

        ax.set_xlabel("KV cache ratio (kept fraction)")
        ax.set_ylabel("token-F1 (MuSiQue)")
        ax.set_title(f"{d['model'].split('/')[-1]}  (N={d['N']}, reduce={d.get('reduce','mean')})")
        ax.invert_xaxis()                      # 1.0 at left → 0.1 at right
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("CompBlend goal-1 (only-HKVD): F1 vs KV-cache ratio across recompute ratios", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
