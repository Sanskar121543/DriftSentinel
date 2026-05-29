"""
SHAP Waterfall Plot Renderer

Generates SHAP waterfall plots as PNG images for consumption by the
multimodal LLM diagnosis agent.  Also generates drift comparison plots
showing reference vs. current SHAP distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_shap_waterfall(
    feature_names: list[str],
    ref_shap_values: list[float],
    cur_shap_values: list[float],
    model_id: str,
    output_dir: Path,
    top_n: int = 15,
) -> list[Path]:
    """
    Render a side-by-side SHAP importance drift comparison plot.

    Shows reference (blue) vs. current (red) mean |SHAP| values for the
    top N most drifted features, sorted by absolute delta.

    Returns list of generated PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(feature_names) != len(ref_shap_values) or len(feature_names) != len(cur_shap_values):
        logger.warning("shap_dimension_mismatch", model_id=model_id)
        return []

    ref_arr = np.array(ref_shap_values)
    cur_arr = np.array(cur_shap_values)
    deltas = np.abs(cur_arr - ref_arr)

    # Select top N by drift magnitude
    top_idx = np.argsort(deltas)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_ref = ref_arr[top_idx]
    top_cur = cur_arr[top_idx]
    top_deltas = deltas[top_idx]

    paths: list[Path] = []

    # -- Plot 1: Drift comparison bar chart --
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_features) * 0.45)))
    y = np.arange(len(top_features))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, top_ref, bar_h, label="Reference", color="#4C72B0", alpha=0.85)
    ax.barh(y - bar_h / 2, top_cur, bar_h, label="Current", color="#DD8452", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        f"SHAP Feature Importance Drift — {model_id}\nTop {len(top_features)} drifted features",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Annotate delta
    for i, (r, c, d) in enumerate(zip(top_ref, top_cur, top_deltas)):
        color = "#c0392b" if d > 0.15 else "#e67e22" if d > 0.05 else "#27ae60"
        ax.text(
            max(r, c) + 0.001,
            i,
            f"Δ{d:.3f}",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold" if d > 0.15 else "normal",
        )

    plt.tight_layout()
    p1 = output_dir / f"{model_id}_shap_drift_comparison.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # -- Plot 2: Δ waterfall for top features --
    fig2, ax2 = plt.subplots(figsize=(10, max(5, len(top_features) * 0.4)))
    colors = ["#c0392b" if d > 0 else "#2980b9" for d in (top_cur - top_ref)]
    ax2.barh(y, top_cur - top_ref, color=colors, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.set_xlabel("SHAP Δ (current − reference)", fontsize=10)
    ax2.set_title(
        f"SHAP Importance Delta — {model_id}\nPositive = feature gained importance, Negative = lost importance",
        fontsize=11,
        fontweight="bold",
    )
    ax2.axvline(x=0, color="black", linewidth=1)
    ax2.invert_yaxis()

    red_patch = mpatches.Patch(color="#c0392b", label="Gained importance (↑)")
    blue_patch = mpatches.Patch(color="#2980b9", label="Lost importance (↓)")
    ax2.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    p2 = output_dir / f"{model_id}_shap_delta_waterfall.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    paths.append(p2)

    logger.info("shap_plots_generated", model_id=model_id, plots=len(paths))
    return paths


def render_feature_distribution_comparison(
    feature_name: str,
    model_id: str,
    ref_histogram_edges: list[float],
    ref_histogram_counts: list[int],
    cur_histogram_counts: list[int],
    output_dir: Path,
) -> Path | None:
    """Render reference vs. current distribution overlay for a single feature."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ref_histogram_edges or len(ref_histogram_counts) != len(cur_histogram_counts):
        return None

    edges = np.array(ref_histogram_edges)
    ref_counts = np.array(ref_histogram_counts, dtype=float)
    cur_counts = np.array(cur_histogram_counts, dtype=float)

    ref_density = ref_counts / (ref_counts.sum() + 1e-9)
    cur_density = cur_counts / (cur_counts.sum() + 1e-9)
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(edges[:-1], ref_density, width=widths, alpha=0.5, label="Reference", color="#4C72B0", align="edge")
    ax.bar(edges[:-1], cur_density, width=widths, alpha=0.5, label="Current", color="#DD8452", align="edge")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution Shift: {feature_name} — {model_id}", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = output_dir / f"{model_id}_{feature_name}_dist_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
