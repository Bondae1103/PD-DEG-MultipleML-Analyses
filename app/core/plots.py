"""
app/core/plots.py
-----------------
Verbatim figure-returning plotting functions ported from notebook cells 12-16.
Follows Plan v2 Section 4.8. Contains zero Streamlit imports and never calls plt.show().
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import shap

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    panel_size: int
) -> plt.Figure:
    """
    Plots the Leave-One-Out Cross-Validation ROC curve matching notebook Cell 12.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4F46E5", lw=2, label=f"Random Forest (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(f"LOOCV ROC — PD Classification\n({panel_size}-Gene Consensus Panel)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    accuracy: float,
    display_labels: List[str]
) -> plt.Figure:
    """
    Plots the LOOCV Confusion Matrix matching notebook Cell 13.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"LOOCV Confusion Matrix\n(Accuracy: {accuracy:.1%})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_gene_boxplots(
    X_final: pd.DataFrame,
    y_raw: pd.Series,
    id_to_symbol: Dict[str, str]
) -> plt.Figure:
    """
    Plots per-gene expression distributions by group (Boxplot + Stripplot) matching notebook Cell 14.
    Dynamically wraps onto a multi-row grid based on consensus panel size.
    """
    plot_df = X_final.copy()
    plot_df.columns = [id_to_symbol.get(str(g), str(g)) for g in plot_df.columns]
    plot_df["STATUS"] = y_raw.values

    gene_cols = [c for c in plot_df.columns if c != "STATUS"]
    n_genes = len(gene_cols)

    ncols = min(n_genes, 6)
    nrows = (n_genes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows), squeeze=False)

    palette = {"NO_PD": "#4C72B0", "PD": "#DD8452"}
    # Handle unexpected status labels
    for g_val in plot_df["STATUS"].unique():
        if g_val not in palette:
            palette[g_val] = "#7F7F7F"

    for idx, gene in enumerate(gene_cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sns.boxplot(data=plot_df, x="STATUS", y=gene, ax=ax, showfliers=False, hue="STATUS", legend=False, palette=palette)
        sns.stripplot(data=plot_df, x="STATUS", y=gene, ax=ax, color="black", alpha=0.4, size=3.5, jitter=0.2)
        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("log2(CPM + 1)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Clean up empty grid slots
    for idx in range(n_genes, nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r, c])

    fig.tight_layout()
    return fig

def plot_clustered_heatmap(
    X_final: pd.DataFrame,
    y_raw: pd.Series,
    id_to_symbol: Dict[str, str],
    class_labels: List[str]
) -> plt.Figure:
    """
    Plots hierarchical clustered heatmap of panel genes across samples matching notebook Cell 15.
    Z-scored per gene across samples.
    """
    heat_df = X_final.copy()
    heat_df.columns = [id_to_symbol.get(str(g), str(g)) for g in heat_df.columns]

    palette = {class_labels[0]: "#4C72B0", class_labels[1]: "#DD8452"}
    col_colors = y_raw.map(palette)

    # Single-gene panel guard: 1D heatmap cannot perform hierarchical clustering on rows
    if len(heat_df.columns) == 1:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(heat_df.T, cmap="vlag", ax=ax, xticklabels=False, cbar_kws={"label": "log2(CPM+1)"})
        ax.set_title(f"Expression Heatmap ({heat_df.columns[0]})", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    g = sns.clustermap(
        heat_df.T,              # genes as rows, samples as columns
        col_colors=col_colors.values,
        cmap="vlag",
        z_score=0,               # z-score across samples per gene
        figsize=(10, max(4.0, len(heat_df.columns) * 0.35 + 2.0)),
        xticklabels=False
    )
    g.figure.suptitle(f"Hierarchical Clustered Heatmap ({len(heat_df.columns)}-Gene Panel)", y=1.02, fontsize=11, fontweight="bold")
    return g.figure

def plot_shap_summary(
    rf_fitted,
    X_final: pd.DataFrame,
    id_to_symbol: Dict[str, str]
) -> plt.Figure:
    """
    Plots the SHAP beeswarm summary plot on the final validated RF matching notebook Cell 16.
    """
    explainer = shap.TreeExplainer(rf_fitted)
    shap_values = explainer.shap_values(X_final)

    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    feature_names = [id_to_symbol.get(str(g), str(g)) for g in X_final.columns]

    fig = plt.figure(figsize=(7.5, max(4.0, len(feature_names) * 0.35 + 2.0)))
    shap.summary_plot(
        vals,
        X_final.values,
        feature_names=feature_names,
        show=False
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig
