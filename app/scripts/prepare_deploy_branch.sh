#!/bin/bash
# Prepares a clean deployment branch for Streamlit Community Cloud or Hugging Face Spaces

set -e

echo "=== Preparing deployment branch: streamlit-deploy ==="

# 1. Ensure git is initialized and changes are staged
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# 2. Checkout or create streamlit-deploy branch
git checkout -B streamlit-deploy

# 3. Ensure requirements.txt matches requirements-app.txt
cp requirements-app.txt requirements.txt

# 4. Stage deployment files
git add streamlit_app.py requirements.txt requirements-app.txt .streamlit/ app/

echo "Deployment branch 'streamlit-deploy' is prepared."
echo "To publish, commit and push to your remote repository:"
echo "  git commit -m 'Deploy Streamlit Biomarker Explorer'"
echo "  git push -u origin streamlit-deploy"
