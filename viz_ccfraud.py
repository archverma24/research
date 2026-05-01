"""
Publication-quality visualizations from CCFraud experimental results.
Writes PNG and PDF figures under figures_ccfraud/.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIG_DIR = Path("figures_ccfraud")
FIG_DIR.mkdir(exist_ok=True)

with open("results/ccfraud_results.json") as f:
    R = json.load(f)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Fig 1: Accuracy Comparison
# ============================================================
def fig_accuracy():
    names = list(R["accuracy"].keys())
    short = ["LR", "RF", "XGB", "SNN", "SNN+ME"]
    means = [R["accuracy"][n]["mean"] for n in names]
    stds = [R["accuracy"][n]["std"] for n in names]
    colors = ["#5B9BD5", "#70AD47", "#FFC000", "#ED7D31", "#C00000"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(short, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw=dict(lw=1.2))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(88, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy Comparison (5-fold)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", alpha=0.3)
    save(fig, "accuracy_comparison")


# ============================================================
# Fig 2: Detailed Metrics Heatmap
# ============================================================
def fig_heatmap():
    names = list(R["metrics"].keys())
    short = ["LR", "RF", "XGB", "SNN", "SNN+ME"]
    met_keys = ["prec", "rec", "f1", "auc"]
    met_labels = ["Precision", "Recall", "F1-Score", "AUC"]

    data = np.array([[R["metrics"][n][k] for k in met_keys] for n in names])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=60, vmax=100, aspect="auto")
    ax.set_xticks(range(len(met_labels)))
    ax.set_xticklabels(met_labels)
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(met_keys)):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if data[i,j] < 75 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score (%)")
    ax.set_title("Performance Metrics Across Models")
    save(fig, "metrics_heatmap")


# ============================================================
# Fig 3: Latency Comparison (log scale)
# ============================================================
def fig_latency():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = ["Edge\n(SNN+ME)", "Cloud\n(Simulated)"]
    vals = [R["latency"]["edge_ms"], R["latency"]["cloud_ms"]]
    errs = [R["latency"]["edge_std"], R["latency"]["cloud_std"]]
    colors = ["#C00000", "#5B9BD5"]

    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.5,
                  error_kw=dict(lw=1.2))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.1,
                f"{v:.2f} ms" if v < 1 else f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Inference Latency (ms, log scale)")
    ax.set_title(f"Edge vs Cloud Latency ({R['latency']['reduction_pct']:.1f}% Reduction)")
    ax.grid(axis="y", alpha=0.3, which="both")
    save(fig, "latency_comparison")


# ============================================================
# Fig 4: Feature Importance (top 15)
# ============================================================
def fig_features():
    fi = R["feature_importance"]
    sel = R["selected_features"]
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0] for x in sorted_fi]
    vals = [x[1] for x in sorted_fi]
    colors = ["#C00000" if n in sel else "#5B9BD5" for n in names]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1],
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("|λ| (Max-Ent Weight)")
    ax.set_title("Feature Importance via Maximum Entropy")
    ax.axvline(x=vals[2], color="gray", linestyle="--", alpha=0.5, label="Selection threshold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "feature_importance")


# ============================================================
# Fig 5: Training Convergence
# ============================================================
def fig_convergence():
    cb = R["convergence_baseline"]
    cp = R["convergence_proposed"]
    epochs = range(1, len(cb)+1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, cb, color="#ED7D31", linewidth=1.5, alpha=0.8, label="SNN Baseline (30 features)")
    ax.plot(epochs, cp, color="#C00000", linewidth=1.5, alpha=0.8, label="SNN + Max-Ent (3 features)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "training_convergence")


# ============================================================
# Fig 6: Radar Chart
# ============================================================
def fig_radar():
    categories = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    models = {
        "XGBoost": [R["accuracy"]["XGBoost"]["mean"],
                    R["metrics"]["XGBoost"]["prec"],
                    R["metrics"]["XGBoost"]["rec"],
                    R["metrics"]["XGBoost"]["f1"],
                    R["metrics"]["XGBoost"]["auc"]],
        "SNN Baseline": [R["accuracy"]["SNN (Baseline)"]["mean"],
                         R["metrics"]["SNN (Baseline)"]["prec"],
                         R["metrics"]["SNN (Baseline)"]["rec"],
                         R["metrics"]["SNN (Baseline)"]["f1"],
                         R["metrics"]["SNN (Baseline)"]["auc"]],
        "SNN+Max-Ent": [R["accuracy"]["SNN + Max-Ent"]["mean"],
                        R["metrics"]["SNN + Max-Ent"]["prec"],
                        R["metrics"]["SNN + Max-Ent"]["rec"],
                        R["metrics"]["SNN + Max-Ent"]["f1"],
                        R["metrics"]["SNN + Max-Ent"]["auc"]],
    }
    colors = {"XGBoost": "#FFC000", "SNN Baseline": "#ED7D31", "SNN+Max-Ent": "#C00000"}

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for name, vals in models.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "o-", linewidth=1.5, label=name, color=colors[name])
        ax.fill(angles, vals_closed, alpha=0.1, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(60, 100)
    ax.set_yticks([70, 80, 90, 100])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Model Comparison Radar", y=1.08)
    save(fig, "radar_comparison")


# ============================================================
# Fig 7: Bandwidth Reduction
# ============================================================
def fig_bandwidth():
    fig, ax = plt.subplots(figsize=(4, 3))
    x = ["Cloud\n(30 features)", "Edge\n(3 features)"]
    vals = [R["bandwidth"]["cloud_bytes"], R["bandwidth"]["edge_bytes"]]
    colors = ["#5B9BD5", "#C00000"]

    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+8,
                f"{v} B", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Payload Size (bytes)")
    ax.set_title(f"Bandwidth Reduction ({R['bandwidth']['reduction_pct']:.1f}%)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "bandwidth_reduction")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig_accuracy()
    fig_heatmap()
    fig_latency()
    fig_features()
    fig_convergence()
    fig_radar()
    fig_bandwidth()
    print("All figures saved to figures_ccfraud/")
