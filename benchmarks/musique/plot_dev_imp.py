"""Plot KV-deviation vs importance profiles from visualize_dev_imp.py JSON.

Usage: python plot_dev_imp.py <devimp.json> <outdir>
Per KV ratio r, makes:
  q1_deviation_kv<r>.png, q1_importance_kv<r>.png, q1_combined_kv<r>.png
  q5_combined_kv<r>.png, q100_combined_kv<r>.png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ratio(rkey, d, bins, outdir):
    tag = f"kv{rkey}"
    xb = (np.arange(bins) + 0.5) / bins

    # ── q1: absolute token position ──
    q1 = d["q1"]
    dev = np.array(q1["dev_raw"]); imp = np.array(q1["imp_raw"])
    bounds = q1["bounds"]; ds0, ds1 = q1["doc_slice"]
    x = np.arange(len(dev))

    def chunk_lines(ax):
        for b in bounds[1:-1]:
            ax.axvline(b, color="gray", lw=0.6, alpha=0.5)
        ax.axvspan(bounds[ds0], bounds[ds1], color="tab:green", alpha=0.05)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, dev, lw=0.7, color="tab:red")
    chunk_lines(ax); ax.set_yscale("log")
    ax.set_xlabel("token position (fused sequence)")
    ax.set_ylabel("KV deviation (layer-mean ‖ΔK‖², log)")
    ax.set_title(f"KV deviation — q#{q1['qidx']}, kv={rkey} (gray=chunk bounds, green=docs)")
    fig.tight_layout(); fig.savefig(f"{outdir}/q1_deviation_{tag}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, imp, lw=0.7, color="tab:blue")
    chunk_lines(ax)
    ax.set_xlabel("token position (fused sequence)"); ax.set_ylabel("importance ((L,H)-mean)")
    ax.set_title(f"KVzip importance — q#{q1['qidx']}, kv={rkey}")
    fig.tight_layout(); fig.savefig(f"{outdir}/q1_importance_{tag}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x, dev, lw=0.7, color="tab:red", label="KV deviation (left, log)")
    ax.set_yscale("log"); ax.set_ylabel("KV deviation", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(x, imp, lw=0.7, color="tab:blue", alpha=0.75, label="importance (right)")
    ax2.set_ylabel("importance", color="tab:blue")
    chunk_lines(ax)
    ax.set_xlabel("token position (fused sequence)")
    ax.set_title(f"deviation vs importance — q#{q1['qidx']}, kv={rkey}")
    fig.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/q1_combined_{tag}.png", dpi=150); plt.close(fig)

    # ── q5 ──
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in d["q5"].values():
        ax.plot(xb, c["dev"], color="tab:red", alpha=0.35, lw=1.0)
        ax.plot(xb, c["imp"], color="tab:blue", alpha=0.35, lw=1.0)
    dev5 = np.mean([c["dev"] for c in d["q5"].values()], axis=0)
    imp5 = np.mean([c["imp"] for c in d["q5"].values()], axis=0)
    ax.plot(xb, dev5, color="tab:red", lw=2.4, label="KV deviation (mean of 5)")
    ax.plot(xb, imp5, color="tab:blue", lw=2.4, label="importance (mean of 5)")
    ax.set_xlabel("relative position within doc chunk (0=start → 1=end)")
    ax.set_ylabel("within-chunk percentile rank (mean)")
    ax.set_title(f"deviation vs importance — 5 random questions, kv={rkey}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/q5_combined_{tag}.png", dpi=150); plt.close(fig)

    # ── q100: mean ± IQR ──
    D = np.array(d["q100"]["dev"]); I = np.array(d["q100"]["imp"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for arr, color, name in ((D, "tab:red", "KV deviation"), (I, "tab:blue", "importance")):
        m = arr.mean(0); q25, q75 = np.percentile(arr, [25, 75], axis=0)
        ax.plot(xb, m, color=color, lw=2.4, label=f"{name} (mean of {len(arr)})")
        ax.fill_between(xb, q25, q75, color=color, alpha=0.18)
    ax.set_xlabel("relative position within doc chunk (0=start → 1=end)")
    ax.set_ylabel("within-chunk percentile rank (mean ± IQR)")
    ax.set_title(f"deviation vs importance — {len(D)} random questions, kv={rkey}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/q100_combined_{tag}.png", dpi=150); plt.close(fig)


def main():
    d = json.load(open(sys.argv[1]))
    outdir = sys.argv[2].rstrip("/")
    for rkey, slot in d["ratios"].items():
        plot_ratio(rkey, slot, d["bins"], outdir)
        print(f"kv={rkey}: 5 figures")
    print(f"done → {outdir}/")


if __name__ == "__main__":
    main()
