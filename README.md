# Sydney Housing Price Prediction and Decision Support System

## Contents
- `Sydney_Housing_ML_Project.ipynb` — full analysis notebook (Parts 1-6)
- `clean_data.py`, `model.py`, `eda.py` — standalone scripts used by the notebook
- `data/raw_suburbs.csv` — original collected dataset (Mosman, Epping, Blacktown)
- `data/cleaned_suburbs.csv` — cleaned/engineered dataset used for modelling
- `data/cv_results.csv`, `data/test_predictions.csv`, `data/part5_comparison.csv` — saved analysis outputs
- `figures/` — exported EDA charts
- `app/streamlit_app.py` — deployed prediction web app
- `app/model.joblib`, `app/model_meta.joblib` — trained model artefacts used by the app

## Running the notebook
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook Sydney_Housing_ML_Project.ipynb
```

## Running the web app
```bash
cd app
pip install streamlit pandas scikit-learn joblib
streamlit run streamlit_app.py
```
Then open http://localhost:8501 in a browser.
