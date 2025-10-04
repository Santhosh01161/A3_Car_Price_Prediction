import os
import numpy as np
import pandas as pd
import pytest
import joblib
import mlflow
import cloudpickle
import warnings
import urllib3

# Suppress SSL and sklearn warnings in CI/CD
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # .../A3 - Predicting Car Prices
# Since all .pkl models are inside "code/", set MODELS_DIR = BASE_DIR
MODELS_DIR = BASE_DIR  
# Cars.csv is also inside "code/"
DATA_PATH = os.path.join(BASE_DIR, "Cars.csv")


# Fixed feature ordering to match a3_model.py
NUMERIC_COLS_ORDER = ['max_power', 'km_driven', 'year']
ALL_FEATURES = NUMERIC_COLS_ORDER


@pytest.fixture(scope="module")
def a3_model():
    mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
    return mlflow.pyfunc.load_model(model_uri="models:/st126107-a3-model/2")

@pytest.fixture(scope="module")
def a3_scaler():
    with open(os.path.join(MODELS_DIR, "cppm_a3_scaler.pkl"), "rb") as f:
        return cloudpickle.load(f)

@pytest.fixture(scope="module")
def sample_data():
    return pd.read_csv(DATA_PATH).iloc[0:1]


# ------------------ A3 Model Tests ------------------
def test_a3_model_load(a3_model, a3_scaler):
    assert a3_model is not None
    assert a3_scaler is not None

def test_a3_model_prediction(a3_model, a3_scaler):
    # Feature definitions must match training
    NUMERIC_COLS_ORDER = ['max_power', 'km_driven', 'year']
    
    ALL_FEATURES = NUMERIC_COLS_ORDER

    # Build sample input (must include all features)
    X_input_dict = {feat: 0 for feat in ALL_FEATURES}
    X_input_dict.update({
        'max_power': 94.5,
        'km_driven': 50000.0,
        'year': 2019,
    })
    X_df = pd.DataFrame([X_input_dict], columns=ALL_FEATURES)

    # --- Scale numeric columns (align names with scaler) ---
    numeric_df = X_df[NUMERIC_COLS_ORDER].copy()
    numeric_df.columns = a3_scaler.feature_names_in_  # align with training names
    X_df[NUMERIC_COLS_ORDER] = a3_scaler.transform(numeric_df)

    # --- Convert to numpy before prediction ---
    X_array = X_df.to_numpy().astype(np.float64)
    pred = a3_model.predict(X_array)

    assert pred is not None
    assert len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number, str, np.str_))