"""Render per-layer (32-panel) deviation vs importance figures.

Usage: python plot_dev_imp_layers.py <devimp_layers.json> <outdir>
Per KV ratio: one PNG with n_layers panels (4 cols). Each panel:
  red  = that layer's KV deviation ||ΔK||² (log scale, left axis)
  blue = that layer's importance (head-mean, right axis)
  gray vertical lines = chunk boundaries (prefix | doc1..docN)
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    d = json.load(open(sys.argv[1]))
    outdir = sys.argv[2].rstrip("/")
    L = d["n_layers"]
    ncol = 4
    nrow = (L + ncol - 1) // ncol

    for rkey, slot in d["ratios"].items():
        dev = np.array(slot["dev"])          # [L, S]
        imp = np.array(slot["imp"])          # [L, S]
        bounds = slot["bounds"]
        S = dev.shape[1]
        x = np.arange(S)

        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 1.9 * nrow),
                                 sharex=True)
        for li in range(L):
            ax = axes[li // ncol][li % ncol]
            dv = np.maximum(dev[li], 1e-6)
            ax.plot(x, dv, lw=0.45, color="tab:red")
            ax.set_yscale("log")
            ax.tick_params(axis="y", labelsize=6, colors="tab:red")
            ax2 = ax.twinx()
            ax2.plot(x, imp[li], lw=0.45, color="tab:blue", alpha=0.65)
            ax2.tick_params(axis="y", labelsize=6, colors="tab:blue")
            for b in bounds[1:-1]:
                ax.axvline(b, color="gray", lw=0.5, alpha=0.45)
            ax.set_title(f"layer {li}", fontsize=8, pad=2)
        for li in range(L, nrow * ncol):
            axes[li // ncol][li % ncol].axis("off")
        fig.suptitle(
            f"per-layer KV deviation (red, log, left) vs importance (blue, right) — "
            f"q#{d['qidx']}, kv={rkey}, prefix+docs (gray = chunk boundaries)",
            fontsize=12)
        fig.supxlabel("token position (fused prefix+docs)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = f"{outdir}/layers_dev_imp_kv{rkey}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
