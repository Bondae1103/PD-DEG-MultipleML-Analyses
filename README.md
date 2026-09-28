# PD-DEG-MultipleML-Analyses

Machine learning identification and cross-validation of transcriptomic biomarker gene panels from differential expression analysis in Parkinson's Disease cortex (GSE68719, GSE136666, GSE168496).

## Pipeline Summary
- **Differential Expression:** `limma-voom` empirical Bayes linear modeling on RNA-Seq read counts.
- **Multi-Method Feature Selection:** Parallel candidate selection via L1-penalized LASSO, Boruta random forests, SVM-RFE, XGBoost importance, and Mutual Information.
- **Validated Biomarker Panel:** 18-gene consensus panel (union of top-10 LASSO and top-10 Boruta) validated under Leave-One-Out Cross-Validation (86.1% LOOCV Accuracy, 0.912–0.939 ROC-AUC).
- **Explainability:** SHAP (`TreeExplainer`) feature attributions quantifying each biomarker's exact mathematical contribution to sample-level predictions.

## Interactive Web Dashboard

A unified, responsive biomarker workbench exploring transcriptome-wide differential expression in Parkinson's disease postmortem cortex, dynamic 5-algorithm consensus voting, published research figure showcases, and explainable patient decision audits.

### Features
1. **Pipeline Architecture:** Interactive 5-stage flowchart detailing sample counts, quality filtering, and biological rationale.
2. **5-Method ML Selection Workbench:** Interactive selection across LASSO, Boruta, SVM-RFE, XGBoost, and Mutual Information with dynamic consensus synthesis (Union, Majority Vote, Strict Intersect).
3. **Custom DEG Upload:** Upload custom candidate DEG CSV/TSV tables or test with preloaded discovery data.
4. **Analysis Plots Showcase:** High-resolution gallery featuring LOOCV ROC curve, Clustered Heatmap, Biomarker Boxplots, Global SHAP Beeswarm summary, and Transcriptome Volcano plot.
5. **Patient Decision Audit:** Live single-sample SHAP waterfall attributions quantifying positive and negative biomarker push.

---

### Deployment Options

#### Option A: Vercel (Fastest & Zero-Build Production)
The app is pre-configured with root `index.html` and `vercel.json` for 10-second deployment on Vercel without server cold starts:
1. Import your repository into [vercel.com](https://vercel.com/new).
2. Leave all default build settings (Framework Preset: `Other`, Build Command: None).
3. Click **Deploy**.

*Or via CLI:*
```bash
npx vercel --prod
```

#### Option B: Local Streamlit (3 Commands)

```bash
# 1. Setup virtual environment
python -m venv .venv-app && source .venv-app/bin/activate

# 2. Install requirements
pip install -r requirements-app.txt

# 3. Launch dashboard
streamlit run streamlit_app.py
```

For detailed cloud deployment steps (including Streamlit Community Cloud and Hugging Face Spaces), see [`app/docs/DEPLOY.md`](app/docs/DEPLOY.md). Technical architecture and verification reports are documented in [`app/docs/`](app/docs/).
