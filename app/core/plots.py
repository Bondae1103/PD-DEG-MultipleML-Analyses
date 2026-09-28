import html
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.config import (
    COLOR_UP, COLOR_DOWN, COLOR_NOT_SIG, COLOR_SHAP_POS, COLOR_SHAP_NEG,
    VOLCANO_LABEL_TOP_N
)

def volcano_figure(
    deg_df: pd.DataFrame,
    p_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    contrast_name: str = "Parkinson's Disease vs Control",
    highlight_genes: Optional[List[str]] = None,
    model_features: Optional[List[str]] = None
) -> go.Figure:
    """
    Creates high-performance interactive Plotly volcano plot using Scattergl.
    """
    fig = go.Figure()

    category_colors = {
        "Up": COLOR_UP,
        "Down": COLOR_DOWN,
        "Not significant": COLOR_NOT_SIG
    }

    # Add a Scattergl trace for each category to allow legend toggling
    for cat in ["Not significant", "Down", "Up"]:
        sub = deg_df[deg_df["category"] == cat]
        if sub.empty:
            continue

        hover_text = [
            f"<b>{html.escape(str(g))}</b><br>"
            f"Category: {cat}<br>"
            f"log2FC: {fc:.2f}<br>"
            f"p-value: {p:.2e}<br>"
            f"FDR padj: {padj:.2e}"
            for g, fc, p, padj in zip(sub["gene"], sub["log2fc"], sub["pvalue"], sub["padj"])
        ]

        fig.add_trace(go.Scattergl(
            x=sub["log2fc"],
            y=sub["neglog10_p"],
            mode="markers",
            name=f"{cat} ({len(sub):,})",
            marker=dict(
                size=5 if cat == "Not significant" else 7,
                color=category_colors[cat],
                opacity=0.6 if cat == "Not significant" else 0.85
            ),
            hoverinfo="text",
            hovertext=hover_text
        ))

    # Highlight model biomarkers with distinct black outline rings
    if model_features:
        model_sub = deg_df[deg_df["gene"].isin(model_features)]
        if not model_sub.empty:
            hover_model = [
                f"<b>{html.escape(str(g))} [Biomarker]</b><br>"
                f"log2FC: {fc:.2f}<br>"
                f"p-value: {p:.2e}<br>"
                f"FDR padj: {padj:.2e}"
                for g, fc, p, padj in zip(model_sub["gene"], model_sub["log2fc"], model_sub["pvalue"], model_sub["padj"])
            ]
            fig.add_trace(go.Scattergl(
                x=model_sub["log2fc"],
                y=model_sub["neglog10_p"],
                mode="markers",
                name="Model Biomarker (18 Genes)",
                marker=dict(
                    size=11,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#000000"),
                    symbol="circle"
                ),
                hoverinfo="text",
                hovertext=hover_model
            ))

    # Add text annotations for top significant genes
    sig_sub = deg_df[deg_df["category"].isin(["Up", "Down"])]
    if not sig_sub.empty:
        top_labels = sig_sub.sort_values("neglog10_p", ascending=False).head(VOLCANO_LABEL_TOP_N)
        for _, row in top_labels.iterrows():
            fig.add_annotation(
                x=row["log2fc"],
                y=row["neglog10_p"],
                text=html.escape(str(row["gene"])),
                showarrow=True,
                arrowhead=1,
                arrowsize=0.8,
                arrowwidth=1,
                arrowcolor="#333333",
                ax=15 if row["log2fc"] > 0 else -15,
                ay=-15,
                font=dict(size=10, color="#111827", family="sans-serif")
            )

    # Threshold guidelines
    neglog_thresh = -np.log10(p_thresh) if p_thresh > 0 else 0
    fig.add_hline(y=neglog_thresh, line_dash="dash", line_color="#6B7280", line_width=1)
    fig.add_vline(x=lfc_thresh, line_dash="dash", line_color="#6B7280", line_width=1)
    fig.add_vline(x=-lfc_thresh, line_dash="dash", line_color="#6B7280", line_width=1)

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title=f"log2 Fold Change ({contrast_name})",
        yaxis_title="-log10(p-value)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="closest"
    )

    return fig

def waterfall_figure(
    frame: pd.DataFrame,
    base_value: float,
    final_margin: float,
    sample_id: str,
    predicted_label: str,
    probability: float
) -> go.Figure:
    """
    Constructs an interactive horizontal SHAP waterfall chart.
    Red bars push prediction toward Parkinson's (positive margin);
    Blue bars push prediction toward Control (negative margin).
    """
    # Measures: absolute for base, relative for features, total for output
    measures = ["absolute"] + ["relative"] * len(frame) + ["total"]
    
    # Gene names: Base value at top, then genes, then final f(x)
    y_labels = ["Base Value E[f(x)]"]
    for _, row in frame.iterrows():
        g = str(row["gene"])
        val = row["feature_value"]
        if pd.notna(val):
            y_labels.append(f"{g} ({val:.2f})")
        else:
            y_labels.append(g)
    y_labels.append("Model Output f(x)")

    # X steps
    x_steps = [base_value] + list(frame["shap"]) + [final_margin]
    
    # Text formatting: signed values
    texts = [f"{base_value:+.2f}"] + [f"{s:+.3f}" for s in frame["shap"]] + [f"{final_margin:+.2f}"]

    # Hover templates
    hovers = [f"Base Value E[f(x)]: {base_value:.3f} log-odds"]
    cum = base_value
    for _, row in frame.iterrows():
        g = str(row["gene"])
        val = row["feature_value"]
        s = float(row["shap"])
        cum += s
        val_str = f"{val:.2f}" if pd.notna(val) else "aggregated"
        direction = "Toward PD" if s > 0 else "Toward Control"
        hovers.append(f"<b>{g}</b><br>Expression: {val_str}<br>SHAP: {s:+.3f} ({direction})<br>Cumulative Margin: {cum:.3f}")
    hovers.append(f"<b>Final f(x)</b>: {final_margin:.3f} log-odds<br>Predicted Probability: {probability*100:.1f}%")

    fig = go.Figure(go.Waterfall(
        name="SHAP Attribution",
        orientation="h",
        measure=measures,
        y=y_labels,
        x=x_steps,
        text=texts,
        textposition="outside",
        hoverinfo="text",
        hovertext=hovers,
        connector={"line": {"color": "#9CA3AF", "width": 1}},
        increasing={"marker": {"color": COLOR_SHAP_POS}},
        decreasing={"marker": {"color": COLOR_SHAP_NEG}},
        totals={"marker": {"color": "#4B5563"}}
    ))

    prob_pct = probability * 100.0
    fig.update_layout(
        title=dict(
            text=f"Sample <b>{sample_id}</b>: Predicted <b>{predicted_label}</b> ({prob_pct:.1f}% probability)",
            font=dict(size=15, color="#111827")
        ),
        xaxis_title="Model Output (Log-Odds Margin)",
        yaxis=dict(autorange="reversed"),  # Largest driver on top
        height=480,
        margin=dict(l=150, r=40, t=50, b=40),
        template="plotly_white",
        showlegend=False
    )

    return fig

def prob_figure(pred_df: pd.DataFrame) -> go.Figure:
    """
    Constructs an interactive probability distribution strip across all samples.
    """
    fig = go.Figure()
    
    colors = [COLOR_SHAP_POS if l == "PD" else COLOR_SHAP_NEG for l in pred_df["predicted_label"]]
    
    fig.add_trace(go.Bar(
        x=pred_df["sample_id"],
        y=pred_df["prob_positive"],
        marker_color=colors,
        text=[f"{p*100:.0f}%" for p in pred_df["prob_positive"]],
        textposition="auto",
        hoverinfo="text",
        hovertext=[
            f"Sample: {s}<br>Predicted: {l}<br>PD Probability: {p*100:.1f}%<br>Margin: {m:.2f}"
            for s, l, p, m in zip(pred_df["sample_id"], pred_df["predicted_label"], pred_df["prob_positive"], pred_df["margin"])
        ]
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="#4B5563", annotation_text="Decision Threshold (0.5)")

    fig.update_layout(
        title="Predicted Probability of Parkinson's Disease Across Cohort",
        xaxis_title="Sample ID",
        yaxis_title="Probability (PD)",
        yaxis=dict(range=[0, 1.05]),
        height=320,
        margin=dict(l=40, r=40, t=40, b=60),
        template="plotly_white"
    )

    return fig
