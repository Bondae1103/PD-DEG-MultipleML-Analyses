# Deployment Guide: PD Transcriptomics Biomarker Explorer

This document provides step-by-step instructions for deploying the interactive dashboard to:
1. **Vercel** (Fastest, zero-build, no cold starts - Recommended)
2. **Streamlit Community Cloud** (Python runtime)
3. **Hugging Face Spaces** (Fallback Python container)

---

## 1. Fast Production Target: Vercel (Zero-Build, Instant CDN)

Vercel provides instant global deployment with **zero cold starts**, zero dependency install errors, and 100% uptime without sleeping. The repository contains a pre-configured root `index.html` and `vercel.json` with high-resolution research figures and client-side consensus computation.

### Option A: 1-Click Import via Vercel Dashboard (Easiest)

1. Push your repository to GitHub.
2. Navigate to [vercel.com/new](https://vercel.com/new) and log in with your GitHub account.
3. Under **"Import Git Repository"**, select your repository (`PD-DEG-MultipleML-Analyses`).
4. In the **"Configure Project"** screen:
   - **Framework Preset:** Leave as `Other` (detected automatically).
   - **Root Directory:** `./` (or leave blank).
   - **Build Command:** Leave blank / none needed.
   - **Output Directory:** Leave blank (serves `index.html` directly from root).
5. Click **"Deploy"**.
6. Deployment finishes in under 10 seconds. You will receive an instant production URL (e.g. `pd-deg-workbench.vercel.app`).

### Option B: Deploy via Command Line (Vercel CLI)

If you have Node.js installed, you can deploy directly from your terminal in 1 command without opening the browser:

```bash
# Deploy to preview
npx vercel

# Deploy directly to production
npx vercel --prod
```

### Features Enabled on Vercel:
- **Interactive Architecture Flowchart:** 5-step visual pipeline with biological rationale and sample counts.
- **5-Method DEG Selection:** Real-time multi-algorithm voting matrix (LASSO, Boruta, SVM-RFE, XGBoost, Mutual Information).
- **Custom DEG Upload:** In-browser CSV/TSV parser supporting custom differential expression tables with automatic format sniffing.
- **Full Research Plot Showcase:** High-resolution interactive gallery (LOOCV ROC curve, Clustered Heatmap, Biomarker Boxplots, SHAP summary beeswarm, Volcano plot).
- **Patient SHAP Audit:** Single-sample waterfall attribution simulation with log-odds margins.

---

## 2. Python Target: Streamlit Community Cloud

### Prerequisites
1. A GitHub account.
2. The repository pushed to GitHub as a public (or private) repository.

### Step-by-Step Deployment
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click the **"New app"** button in the upper right corner.
3. In the deployment modal, configure the following exact settings:
   - **Repository:** `<your-github-username>/<your-repo-name>`
   - **Branch:** `main` (or `streamlit-deploy`)
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Customize if desired (e.g. `pd-biomarker-explorer.streamlit.app`)
4. Click **"Advanced settings..."** (bottom left of the modal):
   - **Python version:** Select **`3.11`** (matches verified runtime environment).
   - **Secrets:** None required (the application runs completely self-contained with no external API keys).
5. Click **"Deploy!"**.
6. Deployment will take approximately 1–2 minutes to install dependencies from `requirements.txt` and launch.

### Important: Keep-Alive Note
> [!NOTE]
> Streamlit Community Cloud automatically puts inactive apps to sleep after consecutive days of inactivity. If presenting to hiring managers or stakeholders, **open the URL 5 minutes beforehand** to ensure cold-start containers are pre-warmed.

---

## 3. Fallback Target: Hugging Face Spaces (Dockerless Streamlit)

If deploying to Hugging Face Spaces:

1. Log into [huggingface.co](https://huggingface.co/) and click **"New Space"**.
2. Enter Space Name (e.g. `pd-biomarker-explorer`).
3. Select **Space SDK:** Choose **`Streamlit`**.
4. Space hardware: Free CPU tier (`2 vCPU, 16 GB RAM`).
5. In your local Git repository or via the Space web editor, ensure the `README.md` front-matter contains:
   ```yaml
   ---
   title: PD Transcriptomics Biomarker Explorer
   emoji: 🧬
   colorFrom: indigo
   colorTo: purple
   sdk: streamlit
   sdk_version: "1.56.0"
   app_file: streamlit_app.py
   pinned: false
   ---
   ```
6. If pushing large Parquet artifacts (>10 MB), track with Git LFS:
   ```bash
   git lfs install
   git lfs track "app/artifacts/*.parquet"
   git add .gitattributes
   ```
7. Push the repository to the Hugging Face Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```

---

## 4. Local Run Instructions

To test or run the application locally on your workstation:

```bash
# 1. Create a clean virtual environment
python -m venv .venv-app

# 2. Activate virtual environment
# Windows PowerShell:
.venv-app\Scripts\Activate.ps1
# Linux / macOS:
source .venv-app/bin/activate

# 3. Install pinned application requirements
pip install -r requirements.txt

# 4. Launch Streamlit dashboard
streamlit run streamlit_app.py
```
The application will open automatically in your browser at `http://localhost:8501`.
