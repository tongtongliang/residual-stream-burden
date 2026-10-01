"""R45 Section 4 figures, using archived covariances, weights and metrics only.

Run with /Users/tongtongliang/miniforge3/envs/ml/bin/python.
All repository paths are relative to this script. No model is loaded or run.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/fig_table/figure/r45"
DATA = ROOT / "paper/fig_table/result_statistic/section03_analysis"
RAW = DATA / "01_patch_embedding_matrix/data"
WHITE = ROOT / "analysis_reports/14_whitening_patch_embedding"
ENDPOINT = DATA / "03_semantic_endpoint_patch_pca"
FID_RAW = ROOT / "paper/fig_table/result_statistic/reference_tables/group1_epoch200.csv"
LABELS = ["Pixel $\\boldsymbol{x}$-pred.", "Pixel $\\boldsymbol{v}$-pred.", "Whitened $\\boldsymbol{x}$-pred.", "Whitened $\\boldsymbol{v}$-pred."]
COLORS = ["#0072B2", "#D55E00", "#56B4E9", "#AA4499"]
STYLES = ["-", "-", "--", ":"]
RANK = np.arange(1, 769)


def relative(path):
    return str(path.relative_to(ROOT))


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def spectral_metrics(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)
    assert values.shape == (768,) and np.all(values > 0)
    values = np.sort(values)[::-1]
    p = values / values.sum()
    return {
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))),
        "stable_rank": float(values.sum() / values[0]),
        "r90": int(np.searchsorted(np.cumsum(p), 0.9) + 1),
        "total_variance": float(values.sum()),
        "largest_eigenvalue": float(values[0]),
    }


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=260,
                    bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6E8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def draw_four(ax, values, cumulative=True):
    for index, (label, color, linestyle, spectrum) in enumerate(
            zip(LABELS, COLORS, STYLES, values)):
        y = np.cumsum(spectrum) / np.sum(spectrum) if cumulative else spectrum
        # The two whitened target spectra coincide exactly. A broad dashed line
        # with a narrower dotted line makes both trace styles visible without
        # shifting data. Alternating markers confirm the same coordinates.
        extra = {}
        if index == 2:
            extra = {"marker": "o", "markevery": [127, 383, 639], "ms": 3.4,
                     "mfc": "white", "mew": 0.75}
        elif index == 3:
            extra = {"marker": "x", "markevery": [255, 511], "ms": 3.8,
                     "mew": 0.9}
        ax.plot(RANK, y, label=label, color=color, ls=linestyle,
                lw=1.8 if index == 2 else 1.35, **extra)
    ax.set(xlim=(1, 768), xticks=[1, 256, 512, 768],
           xlabel="Direction rank")
    if cumulative:
        ax.set(ylim=(0, 1.025), yticks=[0, 0.5, 1], ylabel="Cumulative energy")
    style(ax)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.labelsize": 8.2, "axes.titlesize": 8.8,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    raw = np.load(RAW / "raw_pixel_patch_pca.npz")
    white = np.load(WHITE / "data/embedding_matrices.npz")
    spectrum = np.load(WHITE / "data/spectra.npz")
    matrices = np.load(RAW / "patch_embedding_matrices.npz")
    eigenvalues = np.asarray(raw["eigenvalues"], dtype=float)
    assert np.all(np.diff(eigenvalues) <= 0)
    for suffix in ("basis", "scales", "mean"):
        assert np.array_equal(white[f"clean_{suffix}"], white[f"velocity_{suffix}"])
    assert np.all(white["clean_scales"] > 0)
    targets = [eigenvalues, eigenvalues + 1, np.ones(768), 2 * np.ones(768)]
    assert np.array_equal(np.cumsum(targets[2]) / targets[2].sum(),
                          np.cumsum(targets[3]) / targets[3].sum())
    covariance_definitions = ["C_x", "C_x + I", "I (analytic fitted whitening covariance)",
                              "2 I (analytic whitened clean covariance plus unit Gaussian noise)"]
    raw_fid_rows = rows(FID_RAW)
    raw_fids = [next(row for row in raw_fid_rows if row["run"] == run) for run in
                ("g1_jit_b16_clean_p16_seed0", "g1_jit_b16_velocity_p16_seed0")]
    white_fids = [next(row for row in rows(WHITE / "data/evaluation_history.csv")
                       if row["model"] == model and int(row["checkpoint_step"]) == 250200)
                  for model in ("clean", "velocity")]
    fid_rows = raw_fids + white_fids
    summaries = []
    for index, (label, target, fid_row) in enumerate(zip(LABELS, targets, fid_rows)):
        fid_source = FID_RAW if index < 2 else WHITE / "data/evaluation_history.csv"
        source = RAW / "raw_pixel_patch_pca.npz" if index < 2 else WHITE / "data/embedding_matrices.npz"
        summaries.append({
            "model": label, **spectral_metrics(target), "fid": float(fid_row["fid"]),
            "dimension": 768, "target_covariance": covariance_definitions[index],
            "geometry_source": relative(source), "fid_source": relative(fid_source),
            "geometry_derivation": "cached eigenvalues" if index == 0 else
                "cached clean eigenvalues plus unit isotropic covariance" if index == 1 else
                "analytic covariance implied by full invertible fitted whitening; no new samples",
        })
    with (OUT / "target_geometry_summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary = {
        "rows": summaries,
        "definitions": {
            "effective_rank": "exp(-sum p_i log p_i), p_i=lambda_i/sum(lambda)",
            "stable_rank": "sum(lambda_i)/max(lambda_i)",
            "r90": "minimum number of covariance eigenvalues containing 90 percent total variance",
            "velocity": "v=x-epsilon, with independent epsilon~N(0,I) in training coordinates",
            "whitening": "z=diag(scales)^(-1) basis (x-mean); all 768 positive scales retained",
            "precision": "Whitened target statistics use the analytic fitted covariance I or 2I; saved float32 transform rounding is excluded.",
        },
        "fid_source_rows": fid_rows,
        "geometry_sources": [relative(RAW / "raw_pixel_patch_pca.npz"),
                             relative(WHITE / "data/embedding_matrices.npz"),
                             relative(WHITE / "provenance/pca_estimator.json")],
        "sample_count": {"fit_images": 100000, "fit_patches": 25600000, "patch_dimension": 768},
        "checks": {"positive_whitening_scales": True, "clean_velocity_transforms_identical": True,
                   "whitened_cumulative_curves_identical": True},
    }
    (OUT / "target_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    weight_keys = ["pixel_clean_model_rgb_effective_sv", "pixel_velocity_model_rgb_effective_sv",
                   "whitening_clean_model_whiten_sv", "whitening_velocity_model_whiten_sv"]
    grams = [spectrum[key].astype(float) ** 2 for key in weight_keys]
    _, basis = np.linalg.eigh(raw["covariance"])
    basis = basis[:, ::-1]
    names = list(matrices["model_names"])
    transformed = [matrices[f"weight_{names.index(name)}"].astype(float) @ basis
                   for name in ("JiT-B clean", "JiT-B velocity")]
    transformed += [white[f"whitening_{model}_model_weight"].astype(float)
                    for model in ("clean", "velocity")]
    absolute = [np.abs(matrix) / np.linalg.norm(matrix, "fro") for matrix in transformed]
    norm = LogNorm(vmin=1e-6, vmax=0.1, clip=True)

    fig = plt.figure(figsize=(7.05, 3.8))
    top = fig.add_gridspec(1, 2, left=0.08, right=0.98, bottom=0.60, top=0.885, wspace=0.3)
    a, b = [fig.add_subplot(top[0, index]) for index in range(2)]
    draw_four(a, targets)
    draw_four(b, grams)
    a.set_title("(a) Prediction-target covariance", pad=4)
    b.set_title("(b) Learned embedding Gram", pad=4)
    a.text(0.4, 0.11, "Whitened targets overlap", transform=a.transAxes, fontsize=7, color="#665B68")
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.53, 1), columnspacing=1.25, handlelength=2.4)
    bottom = fig.add_gridspec(1, 4, left=0.08, right=0.927, bottom=0.12, top=0.425, wspace=0.12)
    for index, (label, array) in enumerate(zip(LABELS, absolute)):
        ax = fig.add_subplot(bottom[0, index])
        im = ax.imshow(array, cmap="magma", norm=norm, origin="upper", interpolation="none",
                       extent=(0.5, 768.5, 768.5, 0.5), rasterized=True, aspect="auto")
        ax.set_title(label, color=COLORS[index], fontsize=8, pad=4)
        ax.set(xticks=[1, 384, 768], yticks=[1, 384, 768], xlabel="Patch PCA rank")
        ax.tick_params(length=2, pad=1.5, labelsize=6.6)
        if index == 0:
            ax.set_ylabel("Output channel")
        else:
            ax.tick_params(labelleft=False)
    fig.text(0.08, 0.49, "(c) Learned input maps in patch PCA coordinates", fontsize=8.8)
    cax = fig.add_axes([0.944, 0.12, 0.012, 0.305])
    bar = fig.colorbar(im, cax=cax, ticks=[1e-6, 1e-4, 1e-2])
    bar.ax.tick_params(labelsize=6.5, length=2, pad=1.7)
    fig.text(0.52, 0.008, r"Absolute entries of $WU_x$ (raw) or $W$ (whitened), divided by each map's Frobenius norm",
             ha="center", fontsize=7.5)
    save(fig, "prediction_geometry_embeddings")

    # Separate manuscript variants keep text readable when the spectrum pair
    # occupies only 69 percent of the text width beside a left-hand caption.
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(5.1, 2.1))
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.255, top=0.68,
                            wspace=0.33)
        draw_four(axes[0], targets)
        draw_four(axes[1], grams)
        for ax in axes:
            ax.set(xlabel="Direction rank", xticks=[1, 384, 768])
        axes[0].set_title("(a) Target covariance", pad=5)
        axes[1].set_title("(b) Embedding Gram", pad=5)
        axes[1].set_ylabel("")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                   ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.035),
                   columnspacing=1.9, handlelength=2.1, labelspacing=0.35)
        save(fig, "prediction_spectra_pair")

    with plt.rc_context({"font.size": 9.5, "axes.labelsize": 9.5,
                         "axes.titlesize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9}):
        fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.0))
        fig.subplots_adjust(left=0.078, right=0.92, bottom=0.30, top=0.85,
                            wspace=0.14)
        for index, (ax, label, array) in enumerate(zip(axes, LABELS, absolute)):
            im = ax.imshow(array, cmap="magma", norm=norm, origin="upper",
                           interpolation="none", extent=(0.5, 768.5, 768.5, 0.5),
                           rasterized=True, aspect="auto")
            ax.set_title(label, color=COLORS[index], pad=5)
            ax.set(xticks=[1, 384, 768], yticks=[1, 384, 768], xlabel="Patch PCA rank")
            ax.tick_params(length=2, pad=2)
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
            if index == 0:
                ax.set_ylabel("Output channel")
            else:
                ax.tick_params(labelleft=False)
        cax = fig.add_axes([0.939, 0.30, 0.012, 0.55])
        bar = fig.colorbar(im, cax=cax, ticks=[1e-6, 1e-4, 1e-2])
        bar.ax.tick_params(labelsize=9, length=2, pad=2)
        fig.text(0.53, 0.015,
                 r"$|WU_x|/\|W\|_F$ (raw), $|W|/\|W\|_F$ (whitened); shared log color scale",
                 ha="center", fontsize=9.5)
        save(fig, "embedding_matrices")

    endpoint_rows = rows(ENDPOINT / "all_endpoint_metrics.csv")
    assert len(endpoint_rows) == 27
    model_specs = [("deco_b", "DeCo", "#A07900", "o"),
                   ("pixeldit_b", "PixelDiT", "#009E73", "D"),
                   ("dip_b", "DiP", "#B45A9A", "^")]
    fields = [("endpoint_rank90", "r90", r"$r_{90}$"),
              ("endpoint_effective_rank", "effective_rank", "Effective rank"),
              ("endpoint_stable_rank", "stable_rank", "Stable rank")]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.9))
    fig.subplots_adjust(left=0.069, right=0.993, bottom=0.285, top=0.74, wspace=0.36)
    for ax, (field, baseline_key, label) in zip(axes, fields):
        for slug, name, color, marker in model_specs:
            selected = sorted([row for row in endpoint_rows if row["model"] == slug],
                              key=lambda row: float(row["t_clean"]))
            assert len(selected) == 9
            ax.plot([float(row["t_clean"]) for row in selected],
                    [float(row[field]) for row in selected], label=name,
                    color=color, marker=marker, ms=2.8, lw=1.25)
        for index, text in [(0, "Clean target"), (1, "Velocity target")]:
            ax.axhline(summaries[index][baseline_key], color=COLORS[index],
                       ls="--" if index == 0 else ":", lw=1.15, label=text)
        ax.set(xlim=(0.075, 0.925), xticks=[0.1, 0.5, 0.9],
               xlabel=r"Clean coefficient $t$", ylabel=label, ylim=(0, None))
        style(ax)
    axes[0].set_yticks([0, 300, 600])
    axes[1].set_yticks([0, 150, 300])
    axes[2].set_yticks([0, 4, 8])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.525, 1), columnspacing=1, handlelength=2.1)
    fig.text(0.525, 0.008, "Curves: semantic endpoint states. Horizontal references: target patch covariances.",
             ha="center", fontsize=7.4)
    save(fig, "semantic_endpoints")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.3))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.79, wspace=0.31)
    draw_four(axes[0], targets, cumulative=False)
    axes[0].set(yscale="log", ylabel="Covariance eigenvalue")
    axes[0].set_title("(a) Target patch covariance spectrum", pad=5)
    draw_four(axes[1], targets)
    axes[1].set_title("(b) Cumulative target variance", pad=5)
    axes[1].text(0.41, 0.12, "Whitened curves overlap", transform=axes[1].transAxes,
                 fontsize=7.2, color="#665B68")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.535, 1), columnspacing=1.3, handlelength=2.4)
    save(fig, "target_spectra")

    metadata = {
        "renderer": relative(Path(__file__).resolve()),
        "sources": [relative(path) for path in (
            RAW / "raw_pixel_patch_pca.npz", RAW / "patch_embedding_matrices.npz",
            RAW / "matrix_sources.json", WHITE / "data/embedding_matrices.npz",
            WHITE / "data/spectra.npz", WHITE / "data/evaluation_history.csv",
            WHITE / "provenance/spectrum_sources.json", WHITE / "provenance/pca_estimator.json",
            FID_RAW, ENDPOINT / "all_endpoint_metrics.csv")],
        "counts": {"target_spectra": 4, "directions_per_spectrum": 768,
                   "learned_embedding_matrices": 4, "matrix_shape": [768, 768],
                   "endpoint_models": 3, "times_per_endpoint_model": 9, "endpoint_rows": 27},
        "target_geometry": summary,
        "learned_embedding": {
            "state": "raw epoch200 model weights",
            "spectra": "eigenvalues of native-coordinate weight Gram matrices; cached singular values squared",
            "weight_spectrum_keys": weight_keys,
            "images": "abs(W_raw @ U_raw) or abs(W_whiten_native), each divided by its Frobenius norm",
            "image_axes": "x=descending clean patch PCA direction; y=output channel, original order",
            "image_interpretation": "Learned matrix entries in PCA input coordinates, not eigenvector alignment heatmaps",
            "color_scale": {"normalization": "log", "min": norm.vmin, "max": norm.vmax,
                            "under_range": "shown at the lower colormap endpoint", "shared_between_all_four": True},
            "matrix_absolute_max": [float(array.max()) for array in absolute],
            "matrix_entries_above_color_max": [int((array > norm.vmax).sum()) for array in absolute],
            "matrix_entries_below_color_min": [int((array < norm.vmin).sum()) for array in absolute],
            "matrix_frobenius_norms_before_normalizing": [float(np.linalg.norm(matrix, "fro")) for matrix in transformed],
        },
        "semantic_endpoint": {
            "fields": [field for field, _, _ in fields],
            "time": "x_t=t*x+(1-t)*epsilon; t=1 is clean",
            "point": "silu(t_embedding + semantic_block_output) before separate pixel decoder",
            "protocol": "raw epoch200, true class labels, balanced 100k ImageNet train images, 25.6M patches",
            "baseline": "raw clean/velocity target covariance, not hidden-state endpoints of plain JiT",
            "plotted_rows": endpoint_rows,
        },
        "outputs": [f"{name}.{extension}" for name in
                    ("prediction_geometry_embeddings", "prediction_spectra_pair",
                     "embedding_matrices", "semantic_endpoints", "target_spectra")
                    for extension in ("pdf", "png")],
    }
    (OUT / "section4_r45_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": relative(OUT), "targets": summaries,
                      "endpoint_rows": len(endpoint_rows), "matrix_maxima": metadata["learned_embedding"]["matrix_absolute_max"]}, indent=2))


if __name__ == "__main__":
    main()
