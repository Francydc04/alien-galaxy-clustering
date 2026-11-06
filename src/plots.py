"""Shared chart style and figures for the clustering study."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
METHOD_COLORS = {"K-Means": "#2a78d6", "Hierarchical (Ward)": "#eb6834", "Gaussian mixture": "#1baf7a"}
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=13, color=INK, pad=10)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def model_selection(sweep: pd.DataFrame, bic: pd.DataFrame, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for metric, ax, label in [("silhouette", axes[0], "Silhouette (higher is better)"),
                              ("davies_bouldin", axes[1], "Davies-Bouldin (lower is better)")]:
        for m, c in METHOD_COLORS.items():
            d = sweep[sweep.method == m]
            ax.plot(d.k, d[metric], marker="o", color=c, linewidth=2, label=m)
        style(ax)
        title(ax, label)
        ax.set_xlabel("Number of clusters k", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    axes[2].plot(bic.k, bic.bic, marker="o", color=METHOD_COLORS["Gaussian mixture"], linewidth=2)
    style(axes[2])
    title(axes[2], "Gaussian mixture BIC (lower is better)")
    axes[2].set_xlabel("Number of components", color=MUTED)
    if path:
        save(fig, path)
    return fig


def pca_scatter(coords, labels, explained, names, path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        pts = coords[labels == i]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.55, color=SEGMENT_COLORS[i],
                   edgecolors="none", label=f"{name} (n={len(pts):,})")
    style(ax, grid_axis="both")
    title(ax, "Two segments in principal-component space")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)", color=MUTED)
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)", color=MUTED)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, markerscale=2.5)
    if path:
        save(fig, path)
    return fig


def profile_heatmap(profile_z: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmax = np.abs(profile_z.to_numpy()).max()
    im = ax.imshow(profile_z.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(profile_z.index)), profile_z.index, fontsize=10, color=INK)
    ax.set_xticks(range(len(profile_z.columns)), profile_z.columns, fontsize=10, color=INK)
    for i in range(profile_z.shape[0]):
        for j in range(profile_z.shape[1]):
            v = profile_z.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Segment mean, in standard deviations from the overall mean", color=MUTED, fontsize=9)
    cb.outline.set_visible(False)
    title(ax, "What distinguishes the two segments")
    if path:
        save(fig, path)
    return fig
