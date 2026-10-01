"""Render Section 4 diagnostics from the archived numerical caches only.

Run with /Users/tongtongliang/miniforge3/envs/ml/bin/python. The two depth
figures use identical scales. HyperDiT ends at its actual observed depth.
"""

from pathlib import Path
import csv
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/fig_table/figure/r45"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter


MODELS = [
    ("plain_jit_clean", "Pixel $\\boldsymbol{x}$-pred.", "JiT-B clean", "#0072B2", "o", "-", 12),
    ("plain_jit_velocity", "Pixel $\\boldsymbol{v}$-pred.", "JiT-B velocity", "#D55E00", "s", "-", 12),
    ("deco_b_velocity", "DeCo", "DeCO-B", "#A07900", "^", "--", 12),
    ("pixeldit_b_velocity", "PixelDiT", "PixelDiT-B", "#009E73", "D", "-.", 12),
    ("dip_b_velocity", "DiP", "DiP-B", "#B45A9A", "X", ":", 12),
    ("hyperdit_b_velocity", "HyperDiT", "HyperDiT-B", "#6650A4", "v", "--", 8),
]
WHITENED_MODELS = [
    ("whitening_jit_clean", "Whitened $\\boldsymbol{x}$-pred.", "#56B4E9", "o", "--"),
    ("whitening_jit_velocity", "Whitened $\\boldsymbol{v}$-pred.", "#AA4499", "X", ":"),
]
WHITE_DIAGNOSTIC_MODELS = [(m, label, None, color, marker, style, 12) for m, label, color, marker, style in WHITENED_MODELS]
TIMES = (0.25, 0.5, 0.75)
RATIO_LIMITS = {0.25: 5, 0.5: 12, 0.75: 90}
SENS_PATH = ROOT / "analysis_reports/12_normalized_representation_sensitivity/per_event.csv"
WHITE_SENS_PATH = ROOT / "analysis_reports/12_normalized_representation_sensitivity/whitening_supplement/per_event.csv"
PROBE_PATHS = [
    ROOT / f"analysis_reports/04_hidden_noise_linear_probe/{group}/data/accuracy.csv"
    for group in ("group1", "group1_semantic")
]
WHITE_PROBE_PATH = ROOT / "analysis_reports/04_hidden_noise_linear_probe/group6/data/accuracy.csv"
EMBED_DATA = ROOT / "paper/fig_table/result_statistic/section03_analysis/01_patch_embedding_matrix/data"


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def relative(path):
    return str(path.relative_to(ROOT))


def write_json(name, values):
    (OUT / name).write_text(json.dumps(values, indent=2) + "\n")


def save_figure(fig, stem):
    fig.canvas.draw()
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_diagnostics(models, stem, sensitivity, probes, probe_only_models=()):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12, 3.65))
    fig.subplots_adjust(left=0.077, right=0.99, bottom=0.15,
                        top=0.82, wspace=0.27, hspace=0.32)
    selected_sensitivity, selected_probes, counts, bounds = [], [], [], []
    for column, t in enumerate(TIMES):
        for model, label, _, color, marker, linestyle, depth in models:
            sr = sorted(
                (r for r in sensitivity if r["model"] == model
                 and float(r["t_clean"]) == t
                 and r["point"].startswith("workspace/")
                 and r["point"].endswith("_mlp")),
                key=lambda r: int(r["point"].split("/b")[1].split("_")[0]),
            )
            sx = [int(r["point"].split("/b")[1].split("_")[0]) for r in sr]
            assert sx == list(range(1, depth + 1)), (model, t, sx)
            assert all(int(r["images"]) == 2048 and int(r["noises"]) == 128 for r in sr)
            sy = np.array([float(r["ratio_corrected"]) for r in sr])
            assert np.isfinite(sy).all() and np.all((sy >= 0) & (sy <= RATIO_LIMITS[t]))
            pr = sorted(
                (r for r in probes if r["model"] == model
                 and float(r.get("alpha", r.get("noise_fraction"))) == 1 - t and r["mode"] == "matched_noise"),
                key=lambda r: int(r["after_block"]),
            )
            px = [int(r["after_block"]) for r in pr]
            assert px == list(range(1, depth)), (model, t, px)
            py = np.array([float(r["top1"]) for r in pr])
            sd = np.array([float(r["top1_seed_std"]) for r in pr])
            assert np.isfinite(py).all() and np.isfinite(sd).all() and (sd >= 0).all()
            assert np.all(py - sd >= 0) and np.all(py + sd <= 50)
            style = dict(color=color, marker=marker, ls=linestyle, lw=2.0, ms=5.0)
            axes[0, column].plot(sx, sy, label=label,
                                 markevery=sorted(set(range(0, len(sx), 2)) | {len(sx) - 1}), **style)
            axes[1, column].plot(px, py,
                                 markevery=sorted(set(range(0, len(px), 2)) | {len(px) - 1}), **style)
            axes[1, column].fill_between(px, py - sd, py + sd,
                                         color=color, alpha=0.17, lw=0)
            selected_sensitivity.extend(sr)
            selected_probes.extend(pr)
            counts.append({"model": model, "t_clean": t, "sensitivity_rows": len(sr),
                           "probe_rows": len(pr), "mlp_blocks": sx, "after_blocks": px})
            bounds.append({"model": model, "t_clean": t,
                           "ratio_corrected_min": float(sy.min()),
                           "ratio_corrected_max": float(sy.max()),
                           "probe_lower_with_sd": float((py - sd).min()),
                           "probe_upper_with_sd": float((py + sd).max())})
        for model, label, color, marker, linestyle in probe_only_models:
            pr = sorted((r for r in probes if r["model"] == model
                         and float(r["noise_fraction"]) == 1-t
                         and r["mode"] == "matched_noise"), key=lambda r: int(r["after_block"]))
            px = [int(r["after_block"]) for r in pr]
            assert px == list(range(1,12))
            assert all(float(r["t_clean"]) == t for r in pr)
            py = np.array([float(r["top1"]) for r in pr])
            sd = np.array([float(r["top1_seed_std"]) for r in pr])
            assert np.isfinite(py).all() and np.isfinite(sd).all() and (sd >= 0).all()
            assert (py-sd >= 0).all() and (py+sd <= 50).all()
            axes[1,column].plot(px,py,label=label,color=color,marker=marker,ls=linestyle,
                                lw=2,ms=5,markevery=[0,2,4,6,8,10])
            axes[1,column].fill_between(px,py-sd,py+sd,color=color,alpha=.17,lw=0)
            selected_probes.extend(pr)
            counts.append({"model":model,"t_clean":t,"probe_rows":len(pr),"after_blocks":px})
            bounds.append({"model":model,"t_clean":t,"probe_lower_with_sd":float((py-sd).min()),
                           "probe_upper_with_sd":float((py+sd).max())})
        top, bottom = axes[:, column]
        top.set(title=f"$t={t:.2f}$", ylim=(0, RATIO_LIMITS[t]), xlim=(0.7, 12.3),
                xticks=[1, 6, 12], xlabel="")
        top.tick_params(labelbottom=False)
        top.set_yticks({0.25: [0, 2, 4], 0.5: [0, 6, 12], 0.75: [0, 45, 90]}[t])
        bottom.set(ylim=(0, 50), yticks=[0, 20, 40], xlim=(0.7, 12.3),
                   xticks=[1, 6, 12], xlabel="")
        if column:
            bottom.tick_params(labelleft=False)
        for axis in (top, bottom):
            axis.grid(axis="y", alpha=0.23)
            axis.set_axisbelow(True)
            axis.tick_params(length=3, pad=3)
    axes[0, 0].set_ylabel("Image / noise\n$B/W$", labelpad=3)
    axes[1, 0].set_ylabel("Linear probe\nTop-1 (%)", labelpad=3)
    handles, labels = axes[0,0].get_legend_handles_labels()
    if probe_only_models:
        extra_handles, extra_labels = axes[1,0].get_legend_handles_labels()
        handles += extra_handles
        labels += extra_labels
        axes[0,0].set_ylabel("Raw pixels\n$B/W$", labelpad=3)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.53, 0.982), ncol=len(models)+len(probe_only_models), frameon=False,
               columnspacing=1.05, handlelength=1.9, handletextpad=0.5)
    fig.supxlabel("Residual stream", y=0.01, fontsize=13)
    save_figure(fig, stem)
    provenance = {
        "sensitivity_source": relative(SENS_PATH),
        "whitening_sensitivity_source": relative(WHITE_SENS_PATH) if any(m[0].startswith("whitening") for m in models) else None,
        "probe_sources": [relative(p) for p in PROBE_PATHS] + ([relative(WHITE_PROBE_PATH)] if probe_only_models or any(m[0].startswith("whitening") for m in models) else []),
        "models": [m[0] for m in models], "probe_only_models": [m[0] for m in probe_only_models], "t_clean": list(TIMES), "alpha": "1 - t_clean",
        "sensitivity_metric": "ratio_corrected = between_corrected / within; no clipping",
        "sensitivity_observation": "Main-stream MLP inputs; patch RMS, no spatial pooling",
        "probe_observation": "After each nonfinal main-stream block; patch RMS then spatial average",
        "probe_selection": "matched_noise only; top1 mean and top1_seed_std of validation-noise draws",
        "weights": "raw epoch-200 checkpoints",
        "counts": {"sensitivity_rows": len(selected_sensitivity), "probe_rows": len(selected_probes)},
        "per_model_time_counts": counts, "data_bounds": bounds,
        "axis_bounds": {"ratio_corrected": {str(t): [0, RATIO_LIMITS[t]] for t in TIMES},
                        "probe_including_sd": [0, 50]},
        "checks": "Exact observed block indices, finite values, nonnegative SD, every point and SD band within axes",
        "sensitivity_rows": selected_sensitivity, "probe_rows": selected_probes,
    }
    write_json(f"{stem}_sources.json", provenance)
    return provenance["counts"]


def plot_embedding_filtering():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 0.7,
        "axes.titlesize": 13, "axes.labelsize": 11.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    bins = read_rows(EMBED_DATA / "gain_rank_bins.csv")
    ranks = read_rows(EMBED_DATA / "rankwise_embedding_metrics.csv")
    assert [int(row["rank"]) for row in ranks] == list(range(1, 769))
    with np.load(EMBED_DATA / "raw_pixel_patch_pca.npz") as archive:
        _, clean_vectors = np.linalg.eigh(archive["covariance"])
        clean_vectors = clean_vectors[:, ::-1]
    fig = plt.figure(figsize=(10, 3.8))
    left_edge, panel_width, gap = 0.065, 0.163, 0.019
    heat_height = panel_width * 10 / 3.8
    records, matrix_checks, selected_bins = [], [], []
    with np.load(EMBED_DATA / "patch_embedding_matrices.npz") as archive:
        names = list(archive["model_names"])
        for column, (_, label, name, *_rest) in enumerate([MODELS[1], MODELS[0], *MODELS[2:5]]):
            left = left_edge + column * (panel_width + gap)
            heat = fig.add_axes([left, 0.885 - heat_height, panel_width, heat_height])
            bar = fig.add_axes([left, 0.10, panel_width, 0.18])
            index = names.index(name)
            weight = archive[f"weight_{index}"].astype(np.float64)
            gram = archive[f"gram_{index}"].astype(np.float64)
            # The stored weight maps input columns to output columns; W.T @ W
            # is the input-coordinate Gram matrix used for clean-PCA overlap.
            orientation_error = np.linalg.norm(weight.T @ weight - gram) / np.linalg.norm(gram)
            assert orientation_error < 1e-5, (name, orientation_error)
            _, weight_vectors = np.linalg.eigh(gram)
            weight_vectors = weight_vectors[:, ::-1]
            overlap = (clean_vectors[:, :64].T @ weight_vectors[:, :64]) ** 2
            assert overlap.shape == (64, 64) and np.isfinite(overlap).all()
            assert overlap.min() >= 0 and overlap.max() <= 1 + 1e-12
            image = heat.imshow(overlap.T, origin="lower", cmap="magma",
                                norm=LogNorm(1e-6, 1), extent=[0.5, 64.5, 0.5, 64.5],
                                interpolation="none")
            heat.set(title=label, xticks=[1, 64], yticks=[1, 64], xlabel="Clean-patch rank")
            heat.tick_params(length=2, pad=2)
            heat.get_xticklabels()[0].set_ha("left")
            heat.get_xticklabels()[-1].set_ha("right")
            if column == 0:
                heat.set_ylabel(r"$W_{\mathrm{in}}$ direction rank", labelpad=2)
            else:
                heat.tick_params(labelleft=False)
            gains = np.array([float(row[name + " gain"]) for row in ranks])
            assert np.isfinite(gains).all() and (gains >= 0).all() and gains.sum() > 0
            values = [gains[:64].sum() / gains.sum(), gains[-64:].sum() / gains.sum()]
            for rank_bin, indices, value in zip(("Top 64", "Bottom 64"),
                                                (slice(0, 64), slice(-64, None)), values):
                archived = [row for row in bins if row["model"] == name and row["rank_bin"] == rank_bin]
                assert len(archived) == 1
                assert np.isclose(value, float(archived[0]["gain_share"]), rtol=1e-10, atol=1e-12)
                assert 0 <= value <= 0.9
                selected_bins.extend(archived)
                records.append({"model": name, "rank_bin": rank_bin,
                                "raw_gain_sum": float(gains[indices].sum()),
                                "total_gain": float(gains.sum()), "gain_share": float(value)})
            bar.bar([0, 1], values, width=0.6, color=["#9BC7DF", "#EBC49D"],
                    edgecolor=["#427D9F", "#AB7748"], linewidth=0.6, zorder=3)
            for x, value in enumerate(values):
                text = f"{100 * value:.2f}%" if value < 0.01 else f"{100 * value:.1f}%"
                bar.text(x, value + 0.024, text, ha="center", va="bottom", fontsize=11)
            bar.set(ylim=(0, 0.9), xlim=(-0.55, 1.55), xticks=[0, 1],
                    xticklabels=["Top 64", "Bottom 64"], yticks=[0, 0.4, 0.8])
            bar.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            bar.tick_params(axis="x", length=0, pad=4, labelsize=10.5)
            bar.tick_params(axis="y", length=2, pad=2)
            bar.spines[["top", "right"]].set_visible(False)
            bar.grid(axis="y", color="#DDE2E6", linewidth=0.6, zorder=0)
            if column == 0:
                bar.set_ylabel("Gain share", labelpad=2)
            else:
                bar.tick_params(labelleft=False)
            assert np.allclose([heat.get_position().x0, heat.get_position().width],
                               [bar.get_position().x0, bar.get_position().width])
            matrix_checks.append({"model": name, "input_gram_relative_error": float(orientation_error),
                                  "overlap_min": float(overlap.min()), "overlap_max": float(overlap.max()),
                                  "heatmap_shape": list(overlap.shape)})
    cax = fig.add_axes([0.974, 0.885 - heat_height, 0.010, heat_height])
    fig.colorbar(image, cax=cax, ticks=[1e-6, 1e-3, 1])
    cax.tick_params(labelsize=10.5, length=2, pad=2)
    save_figure(fig, "decoupled_embedding_filtering")
    sources = [EMBED_DATA / name for name in (
        "raw_pixel_patch_pca.npz", "patch_embedding_matrices.npz",
        "rankwise_embedding_metrics.csv", "gain_rank_bins.csv")]
    write_json("decoupled_embedding_filtering_sources.json", {
        "sources": [relative(path) for path in sources],
        "models": [model[2] for model in [MODELS[1], MODELS[0], *MODELS[2:5]]],
        "counts": {"rankwise_rows": len(ranks), "selected_gain_bin_rows": len(selected_bins),
                   "heatmaps": len(MODELS[:5]), "overlap_entries_per_heatmap": 4096},
        "heatmap": "Squared overlap of top-64 eigenvectors. Transposed for display: x clean-patch PCA rank; y input-weight right-singular direction rank. Rank 1 at lower left.",
        "color_scale": {"type": "logarithmic", "floor": 1e-6, "maximum": 1},
        "bars": "Gain sums in ranks 1-64 and 705-768, each divided by the sum over all 768 input directions",
        "checks": "10 recomputed gain shares match archived bins at rtol 1e-10; W.T@W matches input Gram to relative error <1e-5; all bars and heatmap values within axes/color maximum",
        "matrix_checks": matrix_checks, "gain_shares": records, "selected_gain_bin_rows": selected_bins,
    })
    return {"heatmaps": len(MODELS[:5]), "gain_bars": len(records)}


def main():
    sensitivity = read_rows(SENS_PATH)
    white_sensitivity = read_rows(WHITE_SENS_PATH)
    assert len(white_sensitivity) == 144
    for row in white_sensitivity:
        row["source_model"] = row["model"]
        row["model"] = {"whiten_clean": "whitening_jit_clean", "whiten_velocity": "whitening_jit_velocity"}[row["model"]]
    sensitivity += white_sensitivity
    probes = [row for path in PROBE_PATHS + [WHITE_PROBE_PATH] for row in read_rows(path)]
    results = {
        "plain_target_diagnostics": plot_diagnostics(MODELS[:2] + WHITE_DIAGNOSTIC_MODELS, "plain_target_diagnostics", sensitivity, probes),
        "decoupled_diagnostics": plot_diagnostics(MODELS, "decoupled_diagnostics", sensitivity, probes),
        "decoupled_embedding_filtering": plot_embedding_filtering(),
    }
    assert results["plain_target_diagnostics"] == {"sensitivity_rows": 144, "probe_rows": 132}
    assert results["decoupled_diagnostics"] == {"sensitivity_rows": 204, "probe_rows": 186}
    write_json("section4_diagnostics_validation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
