
# Streamlit Portfolio UTBK - prepared files

This repo contains a Streamlit app that predicts UTBK subtest scores per 'JURUSAN/PRODI' using a RandomForest-based multioutput regressor.

Files:
- model_pipeline.pkl : trained pipeline (preprocessing + model)
- nilai_utbk_cleaned.csv : cleaned dataset used for training
- app.py : Streamlit app
- requirements.txt : python dependencies

## How to deploy to GitHub + Streamlit Cloud

1. Download repository files from this environment or push the folder `streamlit-portfolio-utbk` to your GitHub account.
2. Create a new public repo on GitHub and push the contents.
3. Go to https://share.streamlit.io/, sign in with GitHub, click 'New app', connect the repo and select branch `main` and file `app.py`.
4. Click 'Deploy'.
