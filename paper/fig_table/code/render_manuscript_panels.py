"""Compose Section 3 figures from archived arrays/CSVs; no model execution.

Run from any directory: python paper/fig_table/code/render_manuscript_panels.py
Existing standalone figures and all numerical inputs are left unchanged.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result_statistic/section03_analysis"
OUT = ROOT / "figure/manuscript_panels"
MODELS = ["JiT-B velocity", "JiT-B clean", "DeCO-B", "PixelDiT-B", "DiP-B", "HyperDiT-B"]
COLORS = ["#D55E00", "#0072B2", "#C18B00", "#009E73", "#B45A9A", "#56A6CE"]
STYLES = ["--", "-", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def format_axis(ax, rank=True):
    ax.grid(alpha=.18, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.6, pad=2)
    if rank:
        ax.set_xlim(1, 768)
        ax.set_xticks([1, 256, 512, 768])
        ax.set_xlabel("Clean-patch PCA rank")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.2, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    with np.load(DATA / "01_patch_embedding_matrix/data/raw_pixel_patch_pca.npz") as archive:
        eigenvalues = archive["eigenvalues"]
    assert len(eigenvalues) == 768
    ranks = np.arange(1, 769)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.18))
    fig.subplots_adjust(left=.105, right=.99, bottom=.20, top=.90, wspace=.35)
    for t, color, style in zip([1, .75, .5, .25, 0], COLORS, STYLES):
        spectrum = t*t*eigenvalues + (1-t)**2
        axes[0].plot(ranks, spectrum, color=color, ls=style, label=f"$t={t:g}$")
        axes[1].plot(ranks, spectrum.cumsum()/spectrum.sum(), color=color, ls=style)
    for ax in axes:
        format_axis(ax)
    axes[0].set(yscale="log", ylabel="Variance", title="(a) Patch spectrum")
    axes[1].set(ylabel="Cumulative variance", ylim=(0, 1.02), title="(b) Energy concentration")
    axes[0].legend(loc="upper right", frameon=False, fontsize=6.7, labelspacing=.18)
    finish(fig, "patch_geometry")

    rankwise = rows(DATA / "01_patch_embedding_matrix/data/rankwise_embedding_metrics.csv")
    assert len(rankwise) == 768
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65))
    fig.subplots_adjust(left=.105, right=.99, bottom=.32, top=.91, wspace=.35)
    p = eigenvalues/eigenvalues.sum()
    for ax, cumulative in zip(axes, [False, True]):
        ax.plot(ranks, p.cumsum() if cumulative else p, color="black", ls="--", lw=1.1, label="Clean patch energy")
        for model, color, style in zip(MODELS, COLORS, STYLES):
            gain = np.array([float(r[f"{model} gain"]) for r in rankwise])
            gain /= gain.sum()
            ax.plot(ranks, gain.cumsum() if cumulative else gain, color=color, ls=style, label=model)
        format_axis(ax)
    axes[0].set(yscale="log", ylabel="Energy / gain share", title="(a) Directional allocation")
    axes[1].set(ylabel="Cumulative share", ylim=(0, 1.02), title="(b) Cumulative allocation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, -.01), ncol=3, frameon=False, columnspacing=1.1, labelspacing=.35)
    finish(fig, "embedding_filtering")

    aggregate = rows(DATA / "02_residual_stream_sensitivity/data/all_aggregate_metrics.csv")
    intervals = rows(DATA / "02_residual_stream_sensitivity/data/bootstrap_intervals.csv")
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.45), sharey=True)
    fig.subplots_adjust(left=.08, right=.995, bottom=.33, top=.89, wspace=.16)
    for ax, t in zip(axes, [.75, .50, .25]):
        for model, color, style in zip(MODELS, COLORS, STYLES):
            name = model if model.startswith("JiT") else model + " velocity"
            select = lambda r: r["model_display"] == name and abs(float(r["t_clean"])-t) < 1e-7
            values = sorted(filter(select, aggregate), key=lambda r: int(r["point"]))
            bands = sorted(filter(select, intervals), key=lambda r: int(r["point"]))
            assert len(values) == (16 if model == "HyperDiT-B" else 24)
            assert len(bands) == len(values)
            x = [int(r["point"]) for r in values]
            assert x == [int(r["point"]) for r in bands]
            ax.plot(x, [float(r["noise_conditioned_variance_fraction"]) for r in values], color=color, ls=style, label=model)
            ax.fill_between(x, [float(r["bootstrap_low"]) for r in bands], [float(r["bootstrap_high"]) for r in bands], color=color, alpha=.12, lw=0)
        format_axis(ax, rank=False)
        ax.set(xlim=(0, 23), ylim=(0, 1.0), title=f"$t={t:.2f}$", xlabel="Fan-in point")
        ax.set_xticks([0, 5, 11, 17, 23], ["A0", "M2", "M5", "M8", "M11"], rotation=35)
    axes[0].set_ylabel("Noise fraction $F$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, -.01), ncol=3, frameon=False, columnspacing=1.2, labelspacing=.4)
    finish(fig, "residual_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.45), sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.29, top=.89, wspace=.17)
    for ax, slug, name in zip(axes, ["deco_b", "pixeldit_b", "dip_b"], ["DeCo-B", "PixelDiT-B", "DiP-B"]):
        with np.load(DATA / f"03_semantic_endpoint_patch_pca/{slug}_spectra.npz") as archive:
            for i, (t, spectrum) in enumerate(zip(archive["t"], archive["eigenvalues"])):
                ax.plot(ranks, spectrum.cumsum()/spectrum.sum(), color=plt.cm.viridis(i/8), ls=["-", "--", ":"][i%3], label=f"{t:.1f}")
        format_axis(ax)
        ax.set(xlim=(1, 768), ylim=(0, 1.02), title=name, xlabel="Endpoint PCA rank")
        ax.set_xticks([1, 384, 768])
        ax.set_box_aspect(1)
    axes[0].set_ylabel("Cumulative variance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Clean coefficient $t$", title_fontsize=7, loc="lower center", bbox_to_anchor=(.52, -.01), ncol=9, handlelength=1.2, columnspacing=.8, fontsize=6.5, frameon=False)
    finish(fig, "semantic_endpoints")
    residuals = rows(DATA / "04_toy_residual_rank/residual_rank_metrics.csv")
    assert len(residuals) == 50
    fig = plt.figure(figsize=(5.5, 2.90))
    for column, prediction, color in [(0, "clean", "#B9DDF1"), (1, "velocity", "#C6E5B5")]:
        ax = fig.add_axes([column*.50-.01, .02, .49, .88], projection="3d", computed_zorder=False)
        values = [r for r in residuals if r["prediction"] == prediction]
        assert len(values) == 25
        xs, ys, heights = [], [], []
        for r in values:
            x = int(r["block"])
            y = int(round((float(r["t"])-.1)/.2))
            height = int(r["r90"])
            xs.append(x); ys.append(y); heights.append(height)
            ax.text(x, y, height+3, str(height), ha="center", va="bottom", fontsize=6.2, zorder=10)
        xs = np.array(xs); ys = np.array(ys); heights = np.array(heights)
        ax.bar3d(xs-.23, ys-.23, np.zeros_like(heights), .46, .46, heights,
                 color=color, edgecolor="#65747B", linewidth=.3, shade=False, zorder=1)
        ax.set(xlim=(-.6, 4.6), ylim=(4.5, -.5), zlim=(0, 240), xticks=range(5),
               yticks=range(5), yticklabels=[".1", ".3", ".5", ".7", ".9"], zticks=[0, 100, 200])
        ax.set_xlabel("Block", labelpad=-5, fontsize=7)
        ax.set_ylabel("$t$", labelpad=-6, fontsize=8)
        ax.set_zlabel("$r_{90}$", labelpad=-3, fontsize=8)
        ax.set_title(f"FCN-256: {prediction}", pad=1, fontsize=8.5)
        ax.view_init(elev=38, azim=-55)
        ax.set_box_aspect((1.2, 1.4, .9))
        ax.tick_params(labelsize=6, pad=-1)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = (.7, .7, .7, .3)
    finish(fig, "toy_residual_rank")
    print("Rendered five manuscript panels from archived data.")


if __name__ == "__main__":
    main()
