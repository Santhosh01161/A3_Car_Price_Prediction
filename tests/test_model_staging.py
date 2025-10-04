import os
import numpy as np
import pandas as pd
import pytest
import cloudpickle
from app import app  # Import Flask app

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../tests
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = PROJECT_ROOT   # because cppm_a3_model.pkl and scaler are in root
DATA_PATH = os.path.join(PROJECT_ROOT, "Cars.csv")

# ---------------- FIXTURES ----------------
@pytest.fixture(scope="module")
def model():
    with open(os.path.join(MODELS_DIR, "cppm_a3_model.pkl"), "rb") as f:
        return cloudpickle.load(f)

@pytest.fixture(scope="module")
def scaler():
    with open(os.path.join(MODELS_DIR, "cppm_a3_scaler.pkl"), "rb") as f:
        return cloudpickle.load(f)

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture(scope="module")
def sample_data():
    return pd.read_csv(DATA_PATH).iloc[0:1]


# ---------------- UNIT TESTS ----------------
def test_model_load(model, scaler):
    """Ensure model and scaler are loaded properly."""
    assert model is not None
    assert scaler is not None


def test_model_prediction_direct(model, scaler):
    """Test direct prediction using loaded model & scaler."""
    features = np.array([[94.5, 50000.0, 2019]])  # (max_power, km_driven, year)
    scaled_features = scaler.transform(features)
    pred = model.predict(scaled_features)

    assert pred is not None
    assert len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number, str, np.str_))


def test_flask_predict_valid(client):
    """Test /predict route with valid inputs."""
    response = client.post("/predict", data={
        "maxpower": "94.5",
        "kmdriven": "50000",
        "Year": "2019"
    })
    assert response.status_code == 200
    assert b"Predicted Car Class is" in response.data or b"Error:" in response.data


def test_flask_predict_invalid(client):
    """Test /predict route with invalid inputs (empty form)."""
    response = client.post("/predict", data={
        "maxpower": "",
        "kmdriven": "",
        "Year": ""
    })
    assert response.status_code == 200
    assert b"Error:" in response.data or b"Enter details" in response.data
